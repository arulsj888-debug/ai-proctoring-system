
















import base64
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit, disconnect, join_room
from ultralytics import YOLO
import threading
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import logging
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# --- Import Configuration ---
import config

# --- Suppress verbose logging ---
logging.getLogger("ultralytics").setLevel(logging.ERROR)
os.environ['KMP_WARNINGS'] = '0'

# --- Initialization & Model Loading ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key_here!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', path='/YOUR_PATH', logger=False, engineio_logger=False)
mediapipe_lock = threading.Lock()
yolo_lock = threading.Lock()

# --- Smart Database Connection ---
try:
    if hasattr(config, 'MONGO_USER') and config.MONGO_USER:
        print("[DB] Authentication details found. Connecting in authenticated mode.")
        mongo_uri = (f"mongodb://{config.MONGO_USER}:{config.MONGO_PASSWORD}@{config.MONGO_HOST}/?authSource={getattr(config, 'MONGO_AUTH_SOURCE', 'admin')}")
    else:
        print("[DB] No authentication details found. Connecting in local/unauthenticated mode.")
        mongo_uri = f"mongodb://{config.MONGO_HOST}/"
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client[config.MONGO_DB_NAME]
    students_collection = db.students
    print("[DB] Successfully connected to MongoDB.")
except ConnectionFailure as e:
    print(f"[DB FATAL ERROR] Could not connect to MongoDB. Reason: {e}")
    exit()

print("[SERVER START] Initializing MediaPipe models...")
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
print("[SERVER START] MediaPipe models initialized successfully.")

print("[SERVER START] Loading YOLOv8 model...")
try:
    MODEL = YOLO('yolo12n.pt')
    dummy_img = np.zeros((640, 480, 3), dtype=np.uint8)
    MODEL(dummy_img, verbose=False)
    print("[SERVER START] YOLO12N model loaded and ready.")
except Exception as e:
    print(f"[FATAL ERROR] Could not load YOLOv8 model: {e}. The server cannot function.")
    exit()

# --- Constants ---
PROCESSING_INTERVAL = 0.5
EYE_AR_THRESH = 0.20
EYE_CLOSED_SECONDS = 4.0
NO_FACE_SECONDS = 5.0
MOBILE_CONFIDENCE_THRESH = 0.5
STRIKE_LIMIT_MULTIPLE_PERSONS = 2
CRITICAL_WARNING_COOLDOWN = 20
SMTP_SERVER = "YOUR_SMTP_SERVER"
SMTP_PORT = YOUR_SMTP_PORT
SENDER_EMAIL = "YOUR_EMAIL"
SMTP_PASSWORD = "YOUR_PASSWORD"

# --- Data stores ---
user_states = {}
user_warnings = {}
sid_to_user = {}
user_to_sid = {} # NEW: For proctor to target a user

# --- Helper Functions ---
# ... (All helper functions like log_violation_to_db, send_email, create_feedback_email_html, etc. remain the same) ...
# --- [Keep all your existing helper functions here, unchanged] ---
def log_violation_to_db(userId, violation_type, message):
    if not userId or not user_states.get(userId): return
    session_id = user_states[userId].get('session_id')
    if not session_id:
        print(f"[DB LOG] ERROR: No session_id found for user '{userId}'. Cannot log violation.")
        return
    try:
        violation_record = { "timestamp": datetime.now(), "type": violation_type, "details": message }
        students_collection.update_one({"_id": session_id}, {"$push": {"violations": violation_record}})
        print(f"[DB LOG] Logged violation for user '{userId}': {violation_type}")
    except Exception as e:
        print(f"[DB LOG] CRITICAL ERROR: Failed to log violation for user '{userId}'. Reason: {e}")
def send_email(recipient_email, subject, html_body, images_to_attach=None):
    if not all([SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SMTP_PASSWORD]):
        print("[EMAIL SYSTEM] ERROR: SMTP configuration is incomplete.")
        return
    message = MIMEMultipart('related')
    message["Subject"], message["From"], message["To"] = subject, SENDER_EMAIL, recipient_email
    message.attach(MIMEText(html_body, "html"))
    if images_to_attach:
        for cid, image_bytes in images_to_attach.items():
            try:
                image = MIMEImage(image_bytes)
                image.add_header('Content-ID', f'<{cid}>')
                message.attach(image)
            except Exception as e:
                print(f"[EMAIL SYSTEM] ERROR: Failed to attach image with CID {cid}. Reason: {e}")
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(SENDER_EMAIL, SMTP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, message.as_string())
            print(f"[EMAIL SYSTEM] Successfully sent email report to '{recipient_email}'.")
    except Exception as e:
        print(f"[EMAIL SYSTEM] CRITICAL ERROR: Failed to send email to '{recipient_email}'. Reason: {e}")
def create_feedback_email_html(user_id, body_content):
    logo_html = '<div style="text-align:center; margin-top:20px; padding-top:10px; border-top:1px solid #eee;"><img src="cid:logo_image" alt="Logo" style="width:150px;"></div>'
    return f'''<html><body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 20px auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px; border-top: 5px solid #4CAF50;">
            <h2 style="color: #333; border-bottom: 1px solid #eee; padding-bottom: 10px;">Session Feedback for {user_id}</h2>
            <p>Dear {user_id},</p><p>Thank you for completing your proctored session. We have compiled a summary of our automated observations below.</p>
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px;"><h4 style="margin-top: 0; color: #1e3a5f;">Session Summary</h4>{body_content}</div>
            <p style="margin-top: 20px;">Best regards,<br>The Monitoring Team</p>{logo_html}</div></body></html>'''
def create_kickout_email_html(user_id, reason, has_image=False):
    image_html = ''
    if has_image: image_html = '''<p style="margin-top: 20px;"><strong>Evidence Snapshot:</strong></p><div style="text-align: center;"><img src="cid:kickout_evidence" alt="Violation Evidence" style="max-width: 100%; border: 1px solid #ccc; border-radius: 5px;"></div>'''
    logo_html = '<div style="text-align:center; margin-top:20px; padding-top:10px; border-top:1px solid #eee;"><img src="cid:logo_image" alt="Logo" style="width:150px;"></div>'
    return f'''<html><body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 20px auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px; border-top: 5px solid #D32F2F;">
            <h2 style="color: #D32F2F;">Session Terminated</h2><p>Dear {user_id},</p><p>This email is to inform you that your proctoring session was terminated automatically due to a violation of the exam rules.</p>
            <div style="background-color: #FFEBEE; padding: 15px; border-radius: 5px; border-left: 4px solid #D32F2F;"><p style="margin: 0;"><strong>Reason for Termination:</strong><br>{reason}</p></div>{image_html}
            <p style="margin-top: 20px;">Please ensure you adhere to all rules in future sessions. If you believe this was in error, please contact our support team.</p>
            <p>Sincerely,<br>The Monitoring Team</p>{logo_html}</div></body></html>'''
def summarize_timestamps(timestamps, threshold_seconds=5):
    if not timestamps: return "N/A"
    times = sorted(list(set([datetime.strptime(ts, '%H:%M:%S') for ts in timestamps])))
    if not times: return "N/A"
    groups, current_group_start, current_group_end = [], times[0], times[0]
    for i in range(1, len(times)):
        if times[i] <= current_group_end + timedelta(seconds=threshold_seconds): current_group_end = times[i]
        else: groups.append((current_group_start, current_group_end)); current_group_start = current_group_end = times[i]
    groups.append((current_group_start, current_group_end))
    summary_parts = []
    for start, end in groups:
        if (end - start).total_seconds() < 2: summary_parts.append(f"at {start.strftime('%H:%M:%S')}")
        else: summary_parts.append(f"from {start.strftime('%H:%M:%S')} to {end.strftime('%H:%M:%S')}")
    return " and ".join(summary_parts)
def eye_aspect_ratio(eye_landmarks, image_shape):
    p2_y, p6_y = eye_landmarks[1]['y'], eye_landmarks[4]['y']; p3_y, p5_y = eye_landmarks[2]['y'], eye_landmarks[5]['y']; p1_x, p4_x = eye_landmarks[0]['x'], eye_landmarks[3]['x']
    dist_vert1, dist_vert2, dist_horz = abs(p2_y - p6_y), abs(p3_y - p5_y), abs(p1_x - p4_x)
    if dist_horz == 0: return 0.3
    return (dist_vert1 * image_shape[0] + dist_vert2 * image_shape[0]) / (2.0 * dist_horz * image_shape[1])
def bytes_to_image(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[PROCESS] ERROR: Could not decode image bytes. Reason: {e}")
        return None

# --- NEW: Routes for Dashboard ---
@app.route('/dashboard')
def dashboard():
    # In a real application, you would add authentication here
    # to ensure only authorized proctors can access this page.
    return render_template('dashboard.html')

# --- Socket Event Handlers ---
@socketio.on('connect')
def handle_connect(): print(f"[CONNECTION] New client connected. SID: {request.sid}")

@socketio.on('register_user')
def handle_register_user(data):
    sid = request.sid; userId = data.get('userId'); email = data.get('email')
    if not userId or not email: print(f"[REGISTRATION] FAILED: Incomplete data from SID {sid}."); return
    
    # Store mappings
    sid_to_user[sid] = userId
    user_to_sid[userId] = sid

    # ... (Database insertion logic is the same)
    try:
        session_record = { "userId": userId, "email": email, "startTime": datetime.now(), "endTime": None, "status": "ongoing", "violations": [] }
        result = students_collection.insert_one(session_record)
        session_id = result.inserted_id
    except Exception as e:
        # ... (error handling is the same)
        return

    user_states[userId] = { 'session_id': session_id, 'email': email, 'kicked_out': False, 'multiple_person_strikes': 0, 'last_critical_warning_time': 0, 'eye_closed_start_time': None, 'no_face_start_time': None, 'last_warning_time': {}, 'last_processed_time': 0, 'evidence_screenshots': {}, }
    user_warnings[userId] = []
    print(f"[REGISTRATION] SUCCESS: User '{userId}' registered for SID: {sid}")
    
    # Notify proctors about the new student
    socketio.emit('student_joined', {'userId': userId, 'email': email}, room='proctors')
    
    emit('registration_confirmed', {'status': 'success', 'userId': userId})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    userId = sid_to_user.get(sid)
    print(f"[DISCONNECT] Client disconnected. SID: {sid} (User: '{userId if userId else 'Unknown'}')")
    
    if userId and userId in user_states:
        # ... (Email sending and DB finalization logic is the same)
        # Notify proctors that the student has left
        socketio.emit('student_left', {'userId': userId}, room='proctors')
        
        # Cleanup
        if userId in user_to_sid: del user_to_sid[userId]
        # ... (rest of cleanup is the same)

        # [Keep all your existing disconnect logic here]
        session_id = user_states[userId].get('session_id')
        if session_id:
            try:
                final_status = "terminated" if user_states[userId].get('kicked_out', False) else "completed"
                students_collection.update_one( {"_id": session_id}, {"$set": {"endTime": datetime.now(), "status": final_status}} )
            except Exception as e:
                print(f"[DB DISCONNECT] CRITICAL ERROR: Could not finalize session for '{userId}'. Reason: {e}")
        if not user_states[userId].get('kicked_out', False):
            warnings = user_warnings.get(userId, [])
            recipient_email = user_states[userId].get('email')
            evidence_screenshots = user_states[userId].get('evidence_screenshots', {})
            try:
                with open('logo.png', 'rb') as f: logo_data = f.read()
                evidence_screenshots['logo_image'] = logo_data
            except FileNotFoundError:
                print("[EMAIL SYSTEM] WARNING: 'logo.png' not found.")
            if recipient_email:
                email_subject, email_body = generate_feedback_email(userId, warnings, evidence_screenshots)
                html_content = create_feedback_email_html(userId, email_body)
                send_email(recipient_email, email_subject, html_content, images_to_attach=evidence_screenshots)
        if userId in user_states: del user_states[userId]
        if userId in user_warnings: del user_warnings[userId]
    if sid in sid_to_user: del sid_to_user[sid]

# --- NEW: Proctor-specific Socket.IO handlers ---
@socketio.on('join_proctor_room')
def handle_join_proctor_room():
    sid = request.sid
    join_room('proctors')
    print(f"[PROCTOR] Client with SID {sid} joined the 'proctors' room.")

@socketio.on('proctor_action')
def handle_proctor_action(data):
    # Here you would verify if the sender is a real proctor
    action = data.get('action')
    target_user_id = data.get('targetUserId')
    reason = data.get('reason', 'Manual action by proctor.')
    
    target_sid = user_to_sid.get(target_user_id)
    if not target_sid:
        print(f"[PROCTOR ACTION] ERROR: Could not find SID for user '{target_user_id}'.")
        return

    if action == 'terminate':
        print(f"[PROCTOR ACTION] Kicking out user '{target_user_id}' on SID {target_sid}.")
        kickout_user(target_sid, reason)

# --- Modified existing functions to notify proctors ---
def send_warning(sid, warning_type, message, cooldown=5):
    userId = sid_to_user.get(sid)
    if not userId or not user_states.get(userId): return
    state = user_states[userId]
    current_time = time.time()
    last_warning_time = state['last_warning_time'].get(warning_type)
    if last_warning_time and (current_time - last_warning_time < cooldown): return
    
    # Send to student
    emit('monitoring_warning', {'message': message}, room=sid)
    # ALSO send to proctors
    socketio.emit('student_warning_update', {'userId': userId, 'message': message}, room='proctors')
    
    print(f"[WARNING ISSUED] User: '{userId}', Type: '{warning_type}', Message: '{message}'")
    user_warnings[userId].append({'timestamp': current_time, 'type': warning_type, 'message': message})
    state['last_warning_time'][warning_type] = current_time
    log_violation_to_db(userId, warning_type, message)

def kickout_user(sid, reason_message, violation_frame=None):
    userId = sid_to_user.get(sid)
    if not userId or not user_states.get(userId) or user_states[userId].get('kicked_out', False): return
    print(f"!!! [KICKOUT] Kicking out user '{userId}'. Reason: {reason_message} !!!")
    
    log_violation_to_db(userId, "KICKOUT", reason_message)
    user_states[userId]['kicked_out'] = True
    
    # ... (Email sending logic is the same) ...
    # [Keep existing kickout_user logic here for sending email]
    recipient_email = user_states[userId].get('email')
    if recipient_email:
        images_to_attach = {}
        if violation_frame is not None:
            success, buffer = cv2.imencode('.jpg', violation_frame)
            if success: images_to_attach['kickout_evidence'] = buffer.tobytes()
        try:
            with open('logo.png', 'rb') as f: logo_data = f.read()
            images_to_attach['logo_image'] = logo_data
        except FileNotFoundError:
            print("[EMAIL SYSTEM] WARNING: 'logo.png' not found.")
        subject = "Session Terminated Due to Violation"
        html_body = create_kickout_email_html(userId, reason_message, has_image=('kickout_evidence' in images_to_attach))
        send_email(recipient_email, subject, html_body, images_to_attach=images_to_attach)
    
    emit('kickout', {'reason': reason_message}, room=sid)
    socketio.disconnect(sid, silent=True) # This will trigger the handle_disconnect cleanup

@socketio.on('video_frame_binary')
def handle_video_frame_binary(metadata, image_bytes):
    sid = request.sid; userId = sid_to_user.get(sid)
    if not userId or not user_states.get(userId) or user_states[userId].get('kicked_out', False): return
    
    # NEW: Relay frame to proctors in Base64 format
    frame_b64 = base64.b64encode(image_bytes).decode('utf-8')
    socketio.emit('student_video_update', {'userId': userId, 'frame': frame_b64}, room='proctors')
    
    # ... (The rest of the AI processing logic is the same) ...
    # [Keep the rest of your video frame processing logic here]
    state = user_states[userId]
    current_time = time.time()
    if current_time - state['last_processed_time'] < PROCESSING_INTERVAL: return
    state['last_processed_time'] = current_time
    image = bytes_to_image(image_bytes)
    if image is None: return
    with yolo_lock: results = MODEL(image, verbose=False, classes=[67])
    for r in results:
        if len(r.boxes) > 0 and r.boxes.conf[0] > MOBILE_CONFIDENCE_THRESH:
            kickout_user(sid, "Session terminated: Mobile phone detected.", violation_frame=image)
            return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    with mediapipe_lock:
        mesh_results = face_mesh.process(image_rgb)
        detection_results = face_detection.process(image_rgb)
    if detection_results.detections:
        if len(detection_results.detections) > 1: handle_multiple_person_violation(sid, violation_frame=image)
    if mesh_results.multi_face_landmarks:
        state['no_face_start_time'] = None
        face_landmarks = mesh_results.multi_face_landmarks[0]
        LEFT_EYE_LMKS = [362, 385, 387, 263, 373, 380]; RIGHT_EYE_LMKS = [33, 160, 158, 133, 153, 144]
        left_eye = [{'x': face_landmarks.landmark[i].x, 'y': face_landmarks.landmark[i].y} for i in LEFT_EYE_LMKS]
        right_eye = [{'x': face_landmarks.landmark[i].x, 'y': face_landmarks.landmark[i].y} for i in RIGHT_EYE_LMKS]
        avg_ear = (eye_aspect_ratio(left_eye, image.shape) + eye_aspect_ratio(right_eye, image.shape)) / 2.0
        if avg_ear < EYE_AR_THRESH:
            if state['eye_closed_start_time'] is None: state['eye_closed_start_time'] = current_time
            elif current_time - state['eye_closed_start_time'] > EYE_CLOSED_SECONDS:
                send_warning(sid, "EYES_CLOSED", "Eyes have been closed for too long. Please stay attentive.", cooldown=10)
                state['eye_closed_start_time'] = time.time()
        else: state['eye_closed_start_time'] = None
    else:
        if state['no_face_start_time'] is None: state['no_face_start_time'] = current_time
        elif current_time - state['no_face_start_time'] > NO_FACE_SECONDS:
            send_warning(sid, "NO_FACE", "Face not detected. Please ensure you are visible.", cooldown=10)
            state['no_face_start_time'] = time.time()

# --- REWRITTEN EMAIL BODY GENERATION ---
def generate_feedback_email(user_id, warnings, evidence_screenshots):
    subject = f"Feedback for Your Proctoring Session"
    summary = {"EYES_CLOSED": [], "MOBILE_PHONE": [], "MULTIPLE_PERSONS": [], "NO_FACE": []}
    for w in warnings:
        ts = datetime.fromtimestamp(w['timestamp']).strftime('%H:%M:%S')
        if w['type'] in summary: summary[w['type']].append(ts)
    feedback_points = []
    if summary['MULTIPLE_PERSONS']:
        point = f"<strong>Critical Violation:</strong> An additional person was detected at {summarize_timestamps(summary['MULTIPLE_PERSONS'])}."
        if 'MULTIPLE_PERSONS' in evidence_screenshots: point += '<br><p style="margin-left: 20px;"><i>Evidence Snapshot:</i></p><div style="text-align: center; padding: 10px;"><img src="cid:MULTIPLE_PERSONS" style="max-width: 80%; border: 1px solid #ccc;"></div>'
        feedback_points.append(point)
    if len(summary['EYES_CLOSED']) > 0: feedback_points.append(f"<strong>Attention Level:</strong> Prolonged eye closure was noted {summarize_timestamps(summary['EYES_CLOSED'])}.")
    if len(summary['NO_FACE']) > 0: feedback_points.append(f"<strong>Visibility:</strong> Your face was not visible in the frame {summarize_timestamps(summary['NO_FACE'])}.")
    if not feedback_points: body = "<p>We have reviewed your session and are pleased to report that no policy violations were detected. Thank you for maintaining a proper testing environment.</p>"
    else:
        list_items = "".join([f"<li>{point}</li>" for point in feedback_points])
        body = f"""<p>Below is a summary of our observations during your session:</p><ul style="padding-left: 20px; margin-top: 10px;">{list_items}</ul><p>Please review these points to ensure compliance in future sessions.</p>"""
    return subject, body
def handle_multiple_person_violation(sid, violation_frame=None):
    userId = sid_to_user.get(sid)
    if not userId or not user_states.get(userId): return
    state = user_states[userId]
    current_time = time.time()
    if current_time - state.get('last_critical_warning_time', 0) < CRITICAL_WARNING_COOLDOWN: return
    if state['multiple_person_strikes'] == 0 and violation_frame is not None:
        if 'MULTIPLE_PERSONS' not in state['evidence_screenshots']:
            success, buffer = cv2.imencode('.jpg', violation_frame)
            if success:
                state['evidence_screenshots']['MULTIPLE_PERSONS'] = buffer.tobytes()
                print(f"[EVIDENCE] Captured first-offense 'Multiple Persons' screenshot for user '{userId}'.")
    state['multiple_person_strikes'] += 1
    state['last_critical_warning_time'] = current_time
    strike_count = state['multiple_person_strikes']
    if strike_count >= STRIKE_LIMIT_MULTIPLE_PERSONS:
        kickout_user(sid, "Session terminated after multiple warnings for additional people in the frame.", violation_frame=violation_frame)
    else:
        warning_message = f"CRITICAL WARNING ({strike_count}/{STRIKE_LIMIT_MULTIPLE_PERSONS}): Multiple people detected. Ensure you are alone."
        send_warning(sid, "MULTIPLE_PERSONS", warning_message, cooldown=0)

# Main app routes
@app.route('/')
def index(): return jsonify({'status': 'Monitoring server is running', 'port': 2002})
@app.route('/health')
def health(): return jsonify({'status': 'OK', 'active_users': len(user_states), 'connected_clients': len(sid_to_user)})

if __name__ == '__main__':
    print("[SERVER START] Starting Flask-SocketIO server on http://0.0.0.0:YOUR_PORT")
    print("[SERVER START] Socket.IO Path: /iit/socket.io")
    import eventlet
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', YOUR_PORT)), app)
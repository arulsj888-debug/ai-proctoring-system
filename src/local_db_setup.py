import pymongo
from datetime import datetime, timedelta

# --- Configuration for your LOCAL MongoDB instance ---
MONGO_URI = "YOUR_MONGO_HOST"
DB_NAME = "YOUR_DB_NAME"
COLLECTION_NAME = "YOUR_COLLECTION_NAME"

def setup_local_database():
    """
    Connects to a local MongoDB instance, clears the 'students' collection,
    and inserts sample data for testing purposes.
    """
    try:
        # Establish connection to the local server
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info() # Test the connection
        print(f"✅ Successfully connected to local MongoDB at {MONGO_URI}")

        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Clear any existing data in the collection for a fresh start
        print(f"🧹 Clearing old data from the '{COLLECTION_NAME}' collection...")
        collection.delete_many({})
        print("Collection cleared.")

        # --- Create Sample Data ---
        now = datetime.now()

        # Sample 1: A user who completed the session without any violations
        session_good = {
            "userId": "alice_test",
            "email": "alice@example.com",
            "startTime": now - timedelta(minutes=30),
            "endTime": now - timedelta(minutes=5),
            "status": "completed",
            "violations": [] # No violations
        }

        # Sample 2: A user who had a few warnings but completed the session
        session_warnings = {
            "userId": "bob_test",
            "email": "bob@example.com",
            "startTime": now - timedelta(minutes=45),
            "endTime": now - timedelta(minutes=15),
            "status": "completed",
            "violations": [
                {
                    "timestamp": now - timedelta(minutes=40),
                    "type": "NO_FACE",
                    "details": "Face not detected. Please ensure you are visible."
                },
                {
                    "timestamp": now - timedelta(minutes=25),
                    "type": "EYES_CLOSED",
                    "details": "Eyes have been closed for too long. Please stay attentive."
                }
            ]
        }
        
        # Sample 3: A user who was kicked out for a critical violation
        session_terminated = {
            "userId": "charlie_test",
            "email": "charlie@example.com",
            "startTime": now - timedelta(minutes=10),
            "endTime": now - timedelta(minutes=2),
            "status": "terminated",
            "violations": [
                {
                    "timestamp": now - timedelta(minutes=4),
                    "type": "MULTIPLE_PERSONS",
                    "details": "CRITICAL WARNING (1/2): Multiple people detected. Ensure you are alone."
                },
                 {
                    "timestamp": now - timedelta(minutes=2),
                    "type": "KICKOUT",
                    "details": "Session terminated: Mobile phone detected."
                }
            ]
        }

        # Insert the sample documents into the collection
        print("\n🌱 Inserting sample data...")
        collection.insert_many([session_good, session_warnings, session_terminated])
        print(f"👍 Inserted 3 sample session records into '{DB_NAME}.{COLLECTION_NAME}'.")

        # Verify by printing the contents
        print("\n--- Verifying Data in Collection ---")
        for doc in collection.find():
            print(doc)
        print("----------------------------------")
        print("\n🎉 Local database setup is complete!")


    except pymongo.errors.ConnectionFailure:
        print("\n❌ CRITICAL ERROR: Could not connect to MongoDB.")
        print("Please ensure your local MongoDB server (or Docker container) is running on port 27017.")
    finally:
        if 'client' in locals():
            client.close()
            print("\n🔌 Connection to MongoDB closed.")

if __name__ == "__main__":
    setup_local_database()
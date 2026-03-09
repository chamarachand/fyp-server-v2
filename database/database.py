import os
from dotenv import load_dotenv
from google.cloud import firestore

# This looks for a .env file in the current directory and loads it
load_dotenv()

# No need to manually set os.environ if it's already in the .env file,
# but calling it explicitly ensures the SDK sees it immediately.
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize the Async Client
# The SDK will automatically look for the GOOGLE_APPLICATION_CREDENTIALS env var
db = firestore.AsyncClient()

def get_db():
    print("Got db!")
    return db
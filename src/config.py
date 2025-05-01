import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Extract API keys and configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

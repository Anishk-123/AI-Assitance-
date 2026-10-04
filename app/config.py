import os
from dotenv import load_dotenv

load_dotenv()  # reads .env (Groq + LangSmith keys)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

os.makedirs(UPLOAD_DIR, exist_ok=True)

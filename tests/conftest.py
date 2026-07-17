import os


os.environ.setdefault("LLAMA_EMBED_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017/rag_brain_test")


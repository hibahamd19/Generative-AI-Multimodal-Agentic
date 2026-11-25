import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(Path(__file__).parent / '.env')

def model(kind='default'):
    """Fabrique commune des modèles Groq; aucune clé OpenAI requise."""
    default = os.getenv('GROQ_MODEL') or 'qwen/qwen3.8-27b'
    names = {'default': default,
             'fast': os.getenv('GROQ_FAST_MODEL') or 'openai/gpt-oss-20b',
             'vision': os.getenv('GROQ_VISION_MODEL') or 'qwen/qwen3.8-27b'}
    if not os.getenv('GROQ_API_KEY'):
        raise RuntimeError('Renseigner GROQ_API_KEY dans le fichier .env')
    return ChatGroq(model=names[kind], temperature=0, max_tokens=1024, max_retries=2, timeout=60)

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_llm(temperature: float = 0.7):
    return ChatGroq(
        model=os.getenv("MODEL"),
        temperature=temperature,
    )
import os
from openai import OpenAI
import ollama
from dotenv import load_dotenv
from typing import List, Dict

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OLLAMA_API = "http://localhost:11434/api/chat"

openai_client = None

def load_prompts(
    path: str
)-> str:
    with open(path, 'r', encoding="utf-8") as f:
        return f.read()

def get_openai_client():
    load_dotenv()
    v = os.getenv("OPENAI_API_KEY")
    if not v:
        raise RuntimeError("OPENAI API KEY MISSING")
    else:
        openai_client = OpenAI(api_key=v)
    return openai_client

def chat(
    model: str,
    messages: List[Dict],
    temperature: float = 0.2,
    as_json: bool = False
)-> str:
    if model.lower()=="paid":
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            stream=False
        )
        return response.choices[0].message.content.strip()
    elif model.lower()=="free":
        # # payload={
        # #     "model": "llama3.2",
        # #     "messages": messages,
        # #     "stream": False
        # # }
        # response = ollama.chat(
        #     model = "llama3.2",
        #     messages=messages,
        #     stream=False,
        #     format="json"
        # )
        # return response["message"]["content"].strip()
        kwargs = {"model":"llama3.2", "messages":messages}
        if as_json:
            kwargs["format"] = "json"
        response = ollama.chat(**kwargs)
        return response["message"]["content"].strip()
        
    else:
        raise ValueError("Unknown model: please use 'paid' for gpt-4o-mini or 'free' for llama3.2")
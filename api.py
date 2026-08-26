from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from csr import answer_question
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor()

class Message(BaseModel):
    message: str
    history: list[dict] = []

def collect_response(question, history):
    """Runs synchronously in a thread — collects full generator output"""
    final_response = ""
    final_sources = ""
    for response, sources in answer_question(question, history):
        final_response = response
        final_sources = sources
    return final_response, final_sources

@app.post("/chat")
async def chat(msg: Message):

    # Convert plain dicts to LangChain format
    converted_history = []
    for m in msg.history:
        if m.get("role") == "user":
            converted_history.append(HumanMessage(content=m["content"]))
        elif m.get("role") == "assistant":
            converted_history.append(AIMessage(content=m["content"]))

    # Run sync generator in thread so it doesn't block event loop
    loop = asyncio.get_event_loop()
    final_response, final_sources = await loop.run_in_executor(
        executor,
        collect_response,
        msg.message,
        converted_history
    )

    return {
        "response": final_response,
        "sources": final_sources
    }
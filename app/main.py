"""FastAPI application for Qwen chatbot."""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import os
from app.model import get_chatbot
from app.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO if Config.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Qwen Chatbot API",
    description="Chatbot powered by Qwen2.5-72B-Instruct",
    version="1.0.0"
)

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Request/Response models
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = None
    max_length: Optional[int] = None
    top_p: Optional[float] = None


class ChatResponse(BaseModel):
    response: str
    status: str = "success"


# Initialize chatbot (lazy loading)
chatbot = None


@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup."""
    global chatbot
    try:
        logger.info("Initializing chatbot...")
        chatbot = get_chatbot()
        logger.info("Chatbot ready!")
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        chatbot = None


@app.get("/")
async def index():
    """Serve the chat interface."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Chatbot API is running. Please ensure static/index.html exists."}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": chatbot is not None and chatbot.is_loaded() if chatbot else False
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests."""
    if chatbot is None or not chatbot.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Chatbot model is not loaded. Please check server logs."
        )
    
    try:
        # Convert Pydantic models to dicts
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Prepare generation kwargs
        gen_kwargs = {}
        if request.temperature is not None:
            gen_kwargs["temperature"] = request.temperature
        if request.max_length is not None:
            gen_kwargs["max_length"] = request.max_length
        if request.top_p is not None:
            gen_kwargs["top_p"] = request.top_p
        
        # Generate response
        response = chatbot.chat(messages, **gen_kwargs)
        
        return ChatResponse(response=response, status="success")
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG
    )


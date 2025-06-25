# main.py - FastAPI Backend with CrewAI and Ollama
import os 
from typing import List, Dict, Any, Optional, Mapping
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import asyncio
from crewai import Agent, Task, Crew, Process
import ollama
import requests
from langchain.llms.base import LLM

# Get Ollama host from environment variable or use default
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ollama.BASE_URL = OLLAMA_HOST

# Initialize FastAPI application
app = FastAPI(title="CrewAI Chatbot API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Pydantic models for request validation
class Message(BaseModel):
    content: str
    role: str = "user"

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "llama2"  # Default model to use with Ollama

# Ollama client helper
def get_ollama_response(prompt, model="llama2"):
    try:
        response = ollama.chat(model=model, messages=[
            {"role": "user", "content": prompt}
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Ollama error: {e}")
        return f"Error communicating with Ollama: {str(e)}"

# Custom LLM class for Ollama integration with CrewAI
class OllamaLLM(LLM):
    """Custom LLM wrapper for Ollama that works with CrewAI."""
    
    model_name: str = "llama2"
    temperature: float = 0.7
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the Ollama API and return the response."""
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {"role": "user", "content": prompt}
            ], options={"temperature": self.temperature})
            return response['message']['content']
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return f"Error: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "ollama"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters."""
        return {"model_name": self.model_name, "temperature": self.temperature}

# CrewAI setup
def setup_crew(query, model_name="llama2"):
    """Set up a CrewAI crew with proper Ollama integration."""
    
    # Create a custom LLM instance for Ollama
    ollama_llm = OllamaLLM(model_name=model_name, temperature=0.7)
    
    # Create agents with the custom LLM
    researcher = Agent(
        role="Research Specialist",
        goal="Find accurate and relevant information",
        backstory="You are an expert at finding and analyzing information.",
        verbose=True,
        llm=ollama_llm  # Use our custom LLM
    )
    
    writer = Agent(
        role="Content Creator",
        goal="Create engaging and accurate responses",
        backstory="You excel at crafting clear, concise, and helpful content.",
        verbose=True,
        llm=ollama_llm  # Use our custom LLM
    )
    
    # Create tasks
    research_task = Task(
        description=f"Research thoroughly on the query: {query}",
        agent=researcher,
        expected_output="Detailed research findings",
    )
    
    writing_task = Task(
        description="Create a well-crafted response based on the research",
        agent=writer,
        expected_output="Final response to the user query",
        context=[research_task]
    )
    
    # Create the crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True,
        process=Process.sequential
    )
    
    return crew

# REST API endpoints


@app.get("/api/info")
async def get_info():
    """Simple API endpoint that returns information about the API"""
    return {
        "message": "CrewAI Chatbot API is running",
        "version": "1.0.0",
        "models_available": ["llama2", "mistral", "phi", "gemma"]
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Combine all user messages into a single prompt
        user_messages = [msg.content for msg in request.messages if msg.role == "user"]
        combined_prompt = " ".join(user_messages[-3:])  # Use last 3 messages for context
        
        # Get response from Ollama directly (simpler approach)
        response = get_ollama_response(combined_prompt, request.model)
        
        return {
            "response": response,
            "model": request.model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crew_chat")
async def crew_chat(request: ChatRequest):
    try:
        # Extract the latest user message
        latest_message = next((msg.content for msg in reversed(request.messages) 
                              if msg.role == "user"), "")
        
        if not latest_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Setup and run the crew
        crew = setup_crew(latest_message, request.model)
        result = crew.kickoff()
        
        return {
            "response": result,
            "model": request.model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            # Parse the received data
            try:
                request_data = json.loads(data)
                query = request_data.get("message", "")
                model = request_data.get("model", "llama2")
                use_crew = request_data.get("use_crew", False)
                
                # Send acknowledgment
                await manager.send_message(
                    json.dumps({"status": "processing", "message": "Processing your request..."}),
                    websocket
                )
                
                # Process based on whether to use CrewAI or direct Ollama
                if use_crew:
                    # Use background task to avoid blocking
                    crew = setup_crew(query, model)
                    # This will be slow, so we'll send updates
                    await manager.send_message(
                        json.dumps({"status": "update", "message": "CrewAI agents are working on your request..."}),
                        websocket
                    )
                    result = await asyncio.to_thread(crew.kickoff)
                else:
                    # Direct Ollama response
                    result = await asyncio.to_thread(get_ollama_response, query, model)
                
                # Send the final result
                await manager.send_message(
                    json.dumps({
                        "status": "complete",
                        "message": result,
                        "model": model
                    }),
                    websocket
                )
                
            except json.JSONDecodeError:
                await manager.send_message(
                    json.dumps({"status": "error", "message": "Invalid JSON format"}),
                    websocket
                )
            except Exception as e:
                await manager.send_message(
                    json.dumps({"status": "error", "message": str(e)}),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
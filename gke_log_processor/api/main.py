"""FastAPI application for GKE Log Processor."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.config import Config
from ..core.service import LogProcessingService
from ..core.models import AIAnalysisResult

config = Config()
service = LogProcessingService(config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure configuration is valid
    # In a real app, we might check connections here
    yield
    # Shutdown logic if needed

app = FastAPI(
    title="GKE Log Processor API",
    description="API for monitoring and analyzing GKE pod logs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    namespace: str
    pod_name: str
    container: Optional[str] = None
    tail_lines: int = 200

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}

@app.post("/analysis/summary", response_model=AIAnalysisResult)
async def analyze_pod_logs(request: AnalysisRequest):
    """Trigger AI analysis for a pod."""
    logs = await service.get_pod_logs(
        namespace=request.namespace,
        pod_name=request.pod_name,
        container=request.container,
        tail_lines=request.tail_lines
    )
    
    analysis = await service.analyze_logs(logs, analysis_type="summary")
    return analysis

@app.websocket("/ws/logs/{namespace}/{pod_name}")
async def websocket_logs(
    websocket: WebSocket,
    namespace: str,
    pod_name: str,
    container: Optional[str] = None,
    tail_lines: int = 50,
):
    await websocket.accept()
    try:
        # Initial log fetch
        logs = await service.get_pod_logs(
            namespace=namespace,
            pod_name=pod_name,
            container=container,
            tail_lines=tail_lines
        )
        
        # Send initial logs
        for log in logs:
            await websocket.send_json(log.model_dump(mode='json'))
            
        # In a real streaming scenario, we would attach to a k8s watch stream here.
        # For this implementation, we'll simulate streaming or just keep the connection open
        # allowing the client to request updates or we could poll.
        
        # Simple polling simulation for demonstration since we don't have the async watch setup in service yet
        import asyncio
        while True:
            await asyncio.sleep(5)
            # In a real impl, we'd watch for new logs. 
            # Here we just keep the connection alive.
            await websocket.send_json({"type": "ping", "timestamp": str(asyncio.get_event_loop().time())})
            
    except WebSocketDisconnect:
        print(f"Client disconnected: {namespace}/{pod_name}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011)
        except:
            pass

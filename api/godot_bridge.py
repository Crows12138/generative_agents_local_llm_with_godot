"""
Godot-AI Bridge Server
FastAPI server that bridges Godot game engine with AI services
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import json
import traceback

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import AI service components
from ai_service.config_enhanced import get_config, EnhancedConfigManager
from ai_service.ai_service import local_llm_generate
from ai_service.monitoring import get_performance_monitor, get_ai_logger
from ai_service.unified_parser import get_unified_parser
from monitoring.parser_monitor import get_parser_monitor, ParserMonitorContext

# Initialize configuration and monitoring
config_manager = get_config()
config = config_manager.config
perf_monitor = get_performance_monitor()
logger = get_ai_logger()
unified_parser = get_unified_parser()
parser_monitor = get_parser_monitor()

# Create FastAPI app
app = FastAPI(
    title="Godot AI Bridge",
    description="Bridge server for Godot-AI integration",
    version="1.0.0",
    docs_url="/docs" if config.service.enable_docs else None,
    redoc_url="/redoc" if config.service.enable_docs else None
)

# Configure CORS
if config.service.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.service.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request from Godot"""
    character_name: str = Field(..., description="Name of the character")
    message: str = Field(..., description="Message to process")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    max_length: Optional[int] = Field(default=150, description="Maximum response length")
    temperature: Optional[float] = Field(default=0.7, description="Response creativity")

class ChatResponse(BaseModel):
    """Chat response to Godot"""
    character_name: str
    response: str
    emotion: Optional[str] = None
    action: Optional[str] = None
    timestamp: str
    processing_time: float

class DecisionRequest(BaseModel):
    """Decision request from Godot"""
    character_name: str = Field(..., description="Name of the character")
    situation: str = Field(..., description="Current situation description")
    options: List[str] = Field(..., description="Available options")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")

class DecisionResponse(BaseModel):
    """Decision response to Godot"""
    character_name: str
    chosen_option: str
    reasoning: str
    confidence: float
    timestamp: str
    processing_time: float

class ThinkRequest(BaseModel):
    """Think request from Godot"""
    character_name: str = Field(..., description="Name of the character")
    topic: str = Field(..., description="Topic to think about")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")
    depth: Optional[str] = Field(default="normal", description="Thinking depth: quick, normal, deep")

class ThinkResponse(BaseModel):
    """Think response to Godot"""
    character_name: str
    thought: str
    mood: Optional[str] = None
    memory_update: Optional[Dict[str, Any]] = None
    timestamp: str
    processing_time: float

class StatusResponse(BaseModel):
    """Service status response"""
    status: str
    version: str
    active_model: str
    available_models: List[str]
    uptime_seconds: float
    total_requests: int
    average_response_time: float
    health_check: Dict[str, Any]

# Global state
startup_time = datetime.now()
request_count = 0
total_response_time = 0.0

# Helper functions
def format_chat_prompt(character_name: str, message: str, context: Dict[str, Any]) -> str:
    """Format chat prompt for AI model"""
    prompt = f"""You are {character_name}, a character in a game world.
    
Context:
- Location: {context.get('location', 'unknown')}
- Time: {context.get('time', 'unknown')}
- Mood: {context.get('mood', 'neutral')}
- Recent events: {context.get('recent_events', 'none')}

Player says: "{message}"

Respond as {character_name} would, staying in character. Keep your response natural and concise.
Response:"""
    return prompt

def format_decision_prompt(character_name: str, situation: str, options: List[str], context: Dict[str, Any]) -> str:
    """Format decision prompt for AI model"""
    options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
    
    prompt = f"""You are {character_name}, a character in a game world.

Current situation: {situation}

Context:
- Location: {context.get('location', 'unknown')}
- Time: {context.get('time', 'unknown')}
- Goal: {context.get('goal', 'none')}
- Resources: {context.get('resources', 'unknown')}

Available options:
{options_text}

As {character_name}, which option would you choose and why? 
Format your response as:
CHOICE: [option number]
REASONING: [your reasoning]"""
    return prompt

def format_think_prompt(character_name: str, topic: str, context: Dict[str, Any], depth: str) -> str:
    """Format thinking prompt for AI model"""
    depth_instructions = {
        "quick": "Give a brief, immediate thought.",
        "normal": "Provide a thoughtful reflection.",
        "deep": "Provide a detailed, introspective analysis."
    }
    
    prompt = f"""You are {character_name}, a character in a game world.

Topic to think about: {topic}

Context:
- Current activity: {context.get('activity', 'idle')}
- Emotional state: {context.get('emotional_state', 'neutral')}
- Recent interactions: {context.get('recent_interactions', 'none')}
- Personal goals: {context.get('goals', 'none')}

{depth_instructions.get(depth, depth_instructions['normal'])}

Express your inner thoughts as {character_name}:"""
    return prompt

def parse_decision_response(response: str) -> tuple[str, str, float]:
    """Parse decision response from AI"""
    lines = response.strip().split('\n')
    chosen_option = ""
    reasoning = ""
    confidence = 0.8  # default confidence
    
    for line in lines:
        if line.startswith("CHOICE:"):
            chosen_option = line.replace("CHOICE:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
    
    # If parsing fails, use the whole response as reasoning
    if not chosen_option:
        chosen_option = "1"  # default to first option
        reasoning = response
    
    return chosen_option, reasoning, confidence

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Godot AI Bridge",
        "version": "1.0.0",
        "status": "running",
        "docs": f"http://{config.service.host}:{config.service.port}/docs"
    }

@app.post("/ai/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests from Godot"""
    global request_count, total_response_time
    request_count += 1
    
    start_time = datetime.now()
    
    try:
        # Format prompt
        prompt = format_chat_prompt(
            request.character_name,
            request.message,
            request.context or {}
        )
        
        # Call AI service with monitoring
        with ParserMonitorContext(parser_monitor, "chat") as monitor_ctx:
            response = local_llm_generate(
                prompt,
                model_key=None  # Use default active model
            )
            
            # Use unified parser for emotion and action extraction
            emotion = unified_parser.parse_emotion(response)
            action = unified_parser.parse_action(response)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        total_response_time += processing_time
        
        return ChatResponse(
            character_name=request.character_name,
            response=response,
            emotion=emotion,
            action=action,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/decide", response_model=DecisionResponse)
async def decide_endpoint(request: DecisionRequest):
    """Handle decision requests from Godot"""
    global request_count, total_response_time
    request_count += 1
    
    start_time = datetime.now()
    
    try:
        # Format prompt
        prompt = format_decision_prompt(
            request.character_name,
            request.situation,
            request.options,
            request.context or {}
        )
        
        # Call AI service with monitoring
        with ParserMonitorContext(parser_monitor, "decision") as monitor_ctx:
            response = local_llm_generate(prompt, model_key=None)
            
            # Use unified parser for decision parsing
            chosen_option, reasoning, confidence = unified_parser.parse_decision(response, request.options)
        
        # Map option number to actual option text
        try:
            option_idx = int(chosen_option) - 1
            if 0 <= option_idx < len(request.options):
                chosen_option = request.options[option_idx]
            else:
                chosen_option = request.options[0]
        except:
            chosen_option = request.options[0]
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        total_response_time += processing_time
        
        return DecisionResponse(
            character_name=request.character_name,
            chosen_option=chosen_option,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Decide endpoint error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/think", response_model=ThinkResponse)
async def think_endpoint(request: ThinkRequest):
    """Handle thinking requests from Godot"""
    global request_count, total_response_time
    request_count += 1
    
    start_time = datetime.now()
    
    try:
        # Format prompt
        prompt = format_think_prompt(
            request.character_name,
            request.topic,
            request.context or {},
            request.depth
        )
        
        # Adjust parameters based on depth
        max_tokens = {"quick": 100, "normal": 200, "deep": 400}.get(request.depth, 200)
        temperature = {"quick": 0.5, "normal": 0.7, "deep": 0.9}.get(request.depth, 0.7)
        
        # Call AI service with monitoring
        with ParserMonitorContext(parser_monitor, "thinking") as monitor_ctx:
            response = local_llm_generate(prompt, model_key=None)
            
            # Use unified parser for mood extraction
            mood = unified_parser.parse_mood(response)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        total_response_time += processing_time
        
        return ThinkResponse(
            character_name=request.character_name,
            thought=response,
            mood=mood,
            memory_update=None,  # Could be extended to update character memory
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Think endpoint error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/status", response_model=StatusResponse)
async def status_endpoint():
    """Get service status"""
    uptime = (datetime.now() - startup_time).total_seconds()
    avg_response_time = total_response_time / max(request_count, 1)
    
    # Get available models
    available_models = config_manager.list_available_models()
    
    # Get health check
    health_check = {
        "ai_service": "healthy",
        "config_loaded": True,
        "models_available": len(available_models) > 0,
        "memory_usage_mb": perf_monitor.get_memory_usage() if perf_monitor else 0
    }
    
    return StatusResponse(
        status="running",
        version="1.0.0",
        active_model=config.model.active_model,
        available_models=available_models,
        uptime_seconds=uptime,
        total_requests=request_count,
        average_response_time=avg_response_time,
        health_check=health_check
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Godot AI Bridge...")
    
    # Validate configuration
    if not config_manager.validate_config():
        logger.error("Configuration validation failed!")
        
    # Enable hot reload if configured
    if config.monitoring.enable_monitoring:
        config_manager.enable_hot_reload(True)
    
    # Auto-detect models
    detected = config.model.detect_models()
    logger.info(f"Detected {len(detected)} models")
    
    # Warm up AI service
    try:
        logger.info("Warming up AI service...")
        test_response = local_llm_generate("Hello, this is a test.", model_key=None)
        logger.info(f"AI service ready: {test_response[:50]}...")
    except Exception as e:
        logger.error(f"Failed to warm up AI service: {e}")
    
    logger.info(f"Godot AI Bridge started on http://{config.service.host}:{config.service.port}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Godot AI Bridge...")
    
    # Disable hot reload
    config_manager.enable_hot_reload(False)
    
    logger.info("Godot AI Bridge stopped")

# Main entry point
def main():
    """Run the bridge server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Godot AI Bridge Server")
    parser.add_argument("--host", default=config.service.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.service.port, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    # Update config with command line args
    config.service.host = args.host
    config.service.port = args.port
    
    # Run server
    uvicorn.run(
        "api.godot_bridge:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=config.monitoring.log_level.lower()
    )

if __name__ == "__main__":
    main()



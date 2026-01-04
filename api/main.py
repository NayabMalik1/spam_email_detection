"""
FastAPI application for Email Spam Detection ANN
"""

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import yaml
import logging
from pathlib import Path
from datetime import datetime
import json
import sys
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import nltk  # <-- NLTK import hai already

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference.predictor import SpamPredictor
from utils.logger import setup_logger
from api.routers import predict, analysis, dataset, reports
from api.schemas import HealthResponse

# ✅ NLTK SETUP - FIXED FOR WINDOWS
try:
    # Download required NLTK data quietly
    nltk.download('omw-eng', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("[SUCCESS] NLTK data loaded successfully")  # Plain text
except Exception as e:
    print(f"[WARNING] NLTK warning: {e} - Continuing without some NLP features")
    # Continue execution even if NLTK download fails

# Setup logging
logger = setup_logger(__name__)

# Rest of your code remains the same...

# Load configuration
config_path = project_root / 'configs/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Global predictor instance
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    global predictor
    logger.info("Starting up Email Spam Detection API...")
    
    # ✅ OPTIONAL: You can also add NLTK check here
    try:
        from nltk.corpus import stopwords
        _ = stopwords.words('english')
        logger.info("NLTK is properly configured")
    except:
        logger.warning("NLTK stopwords not available, using fallback")
    
    try:
        # Initialize predictor
        predictor = SpamPredictor(config_path)
        if predictor.load_models():
            logger.info("Models loaded successfully")
        else:
            logger.error("Failed to load models")
            raise RuntimeError("Failed to load models")
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down Email Spam Detection API...")
        predictor = None

# Create FastAPI app
app = FastAPI(
    title=config['api']['title'],
    description=config['api']['description'],
    version=config['api']['version'],
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Baaki ka code same rahega...

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = project_root / "api" / "static"
static_dir.mkdir(exist_ok=True, parents=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates_dir = project_root / "api" / "templates"
templates = Jinja2Templates(directory=templates_dir)

# Include routers
app.include_router(predict.router, prefix="/api", tags=["prediction"])
app.include_router(analysis.router, prefix="/api", tags=["analysis"])
app.include_router(dataset.router, prefix="/api", tags=["dataset"])
app.include_router(reports.router, prefix="/api", tags=["reports"])

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "details": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serve the main web interface
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": config['web']['title'],
            "version": config['api']['version']
        }
    )

# Health check endpoint
@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint
    """
    global predictor
    
    status = "healthy"
    details = {}
    
    if predictor is None:
        status = "unhealthy"
        details["error"] = "Predictor not initialized"
    elif not predictor.models_loaded:
        status = "unhealthy"
        details["error"] = "Models not loaded"
    else:
        # Get prediction statistics
        stats = predictor.get_prediction_stats()
        details["prediction_stats"] = stats
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        version=config['api']['version'],
        details=details
    )

# API info endpoint
@app.get("/api/info", tags=["info"])
async def api_info():
    """
    Get API information
    """
    return {
        "name": config['api']['title'],
        "description": config['api']['description'],
        "version": config['api']['version'],
        "models_loaded": predictor.models_loaded if predictor else False,
        "endpoints": [
            {"path": "/api/predict", "method": "POST", "description": "Predict spam for single text"},
            {"path": "/api/predict/batch", "method": "POST", "description": "Predict spam for multiple texts"},
            {"path": "/api/analyze", "method": "POST", "description": "Analyze text with detailed results"},
            {"path": "/api/dataset/info", "method": "GET", "description": "Get dataset information"},
            {"path": "/api/reports/list", "method": "GET", "description": "List available reports"},
            {"path": "/api/health", "method": "GET", "description": "Health check"},
            {"path": "/api/docs", "method": "GET", "description": "API documentation"},
        ],
        "configuration": {
            "model": config['model'],
            "api": {
                "host": config['api']['host'],
                "port": config['api']['port'],
                "workers": config['api']['workers']
            }
        }
    }

# Dependency to get predictor
def get_predictor():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    return predictor

def run_api():
    """
    Run the FastAPI application
    """
    uvicorn.run(
        "api.main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        workers=config['api']['workers'],
        reload=config['api']['debug']
    )

if __name__ == "__main__":
    run_api()
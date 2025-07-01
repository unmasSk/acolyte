"""
ACOLYTE API Module
Basic FastAPI application for the ACOLYTE backend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="ACOLYTE API",
    description="Local AI Programming Assistant API",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Solo localhost en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "acolyte-backend"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "ACOLYTE API is running"}


# TODO: Add actual API endpoints here
# - /v1/chat/completions
# - /v1/embeddings
# - /api/index
# etc.

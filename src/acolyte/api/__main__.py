"""
Main entry point for ACOLYTE API when run with python -m

This allows the API to be run with:
    python -m acolyte.api

Which is what the Docker container uses.
"""

import uvicorn
from acolyte.api import app

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

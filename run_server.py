#!/usr/bin/env python
"""Run the FastAPI server with the correct port from environment variable."""

import os

import uvicorn

from src.api.main import app

if __name__ == "__main__":
    # Render uses PORT environment variable
    port = int(os.getenv("PORT", 10000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,  # Render free tier only supports 1 worker
    )

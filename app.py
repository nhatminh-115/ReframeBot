"""Entry point — run with: python app.py"""
import sys
from pathlib import Path

# Make the src package importable without installing the project
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn
from reframebot.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "reframebot.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )

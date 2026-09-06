"""Run the local operator dashboard with ``python -m src.api``."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "src.api.app:app",
        host=os.getenv("LEMURU_HOST", "127.0.0.1"),
        port=int(os.getenv("LEMURU_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()


import os

import uvicorn

from app import app


def main() -> None:
    host = os.environ.get("PBRAIN_HOST", "127.0.0.1")
    port = int(os.environ.get("PBRAIN_PORT", "8787"))
    log_level = os.environ.get("PBRAIN_LOG_LEVEL", "info")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()

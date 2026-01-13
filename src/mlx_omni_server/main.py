import argparse
import os

import uvicorn
from fastapi import FastAPI

from .middleware.logging import RequestResponseLoggingMiddleware
from .routers import api_router
from .utils.logger import logger, set_logger_level

app = FastAPI(title="MLX Omni Server")

# Add request/response logging middleware with custom levels
app.add_middleware(
    RequestResponseLoggingMiddleware,
    # exclude_paths=["/health"]
)

from fastapi.middleware.cors import CORSMiddleware

app.include_router(api_router)


def build_parser():
    """Create and configure the argument parser for the server."""
    parser = argparse.ArgumentParser(description="MLX Omni Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to, defaults to 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10240,
        help="Port to bind the server to, defaults to 10240",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use, defaults to 1",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level, defaults to info",
    )

    parser.add_argument(
        "--cors-allow-origins",
        type=str,
        default="",
        help='Apply origins to CORSMiddleware. This is useful for accessing the local server directly from the browser (use --cors-allow-origins="*"). Defaults to disabled',
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to custom models directory (org/model-name structure). Takes precedence over HuggingFace cache. Defaults to None",
    )
    return parser


def configure_cors_middleware(cors_allow_origins: str | None):
    """Configure CORS middleware with the provided origins, if any."""
    # Remove existing CORS middleware
    app.user_middleware = [m for m in app.user_middleware if m.cls != CORSMiddleware]
    app.middleware_stack = None  # Reset middleware stack to force rebuild

    if cors_allow_origins is None:
        origins = []
    else:
        # Add CORS middleware with provided origins or empty list
        origins = (
            [origin.strip() for origin in cors_allow_origins.split(",")]
            if cors_allow_origins
            else []
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


configure_cors_middleware(os.environ.get("MLX_OMNI_CORS", None))


def start():
    """Start the MLX Omni Server."""

    parser = build_parser()
    args = parser.parse_args()

    # Set log level through environment variable
    os.environ["MLX_OMNI_LOG_LEVEL"] = args.log_level
    # Set CORS through environment variable
    os.environ["MLX_OMNI_CORS"] = args.cors_allow_origins
    # Set model path through environment variable (CLI arg takes precedence over env var)
    if args.model_path:
        os.environ["MLX_OMNI_MODEL_PATH"] = args.model_path
    elif "MLX_OMNI_MODEL_PATH" not in os.environ:
        # If neither CLI arg nor env var is set, ensure env var is not set
        os.environ.pop("MLX_OMNI_MODEL_PATH", None)

    set_logger_level(logger, args.log_level)
    configure_cors_middleware(args.cors_allow_origins)

    # Log the configured model path
    model_path = os.environ.get("MLX_OMNI_MODEL_PATH")
    if model_path:
        logger.info(f"Custom model path configured: {model_path}")
    else:
        logger.info("Using HuggingFace cache for models")

    # Start server with uvicorn
    uvicorn.run(
        "mlx_omni_server.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        use_colors=True,
        workers=args.workers,
    )


if __name__ == "__main__":
    start()

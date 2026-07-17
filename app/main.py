import os

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.api.router import api_router
from app.core.rate_limiter import limiter
from app.core.database import init_db

load_dotenv()

# Determine whether to expose the interactive documentation.
env = os.getenv("ENVIRONMENT", "development").lower()
docs_url = "/docs"
redoc_url = "/redoc"
openapi_url = "/openapi.json"
if env == "production":
    docs_url = None
    redoc_url = None
    openapi_url = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan hook."""
    await init_db()

    # Keep the lifespan hook because startup still initializes Mongo indexes
    # and it provides a clear extension point for future bootstrapping work.
    yield


app = FastAPI(
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

if env != "production":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix="/api")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return {"message": "API ready!"}

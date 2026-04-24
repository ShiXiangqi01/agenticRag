import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.routing import APIRoute


from src.api.lifespan import api_lifespan  




app = FastAPI(
    title="Agentic API",
    description="The Agentic API",
    lifespan=api_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.add_middleware(GZipMiddleware, minimum_size=500)

# Include routers after app is created
from src.api.routers import (  

    search,
    user_image,
    look_up,
    assistant,
    agent,

)

app.include_router(search.router)
app.include_router(user_image.router)
app.include_router(look_up.router)
app.include_router(assistant.router)
app.include_router(agent.router)



def run_api():
    
    host = os.environ.get("API_HOST", "localhost")
    port = int(os.environ.get("API_PORT", 8000))

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=bool(int(os.environ.get("DEBUG", "0"))),
    )


if __name__ == "__main__":
    run_api()

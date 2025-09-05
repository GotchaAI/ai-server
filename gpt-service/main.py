import uvicorn
from fastapi import FastAPI

from lulu_routes import router as lulu_router
from myomyo_routes import router as myomyo_router

app = FastAPI(
    title="GPT Service",
    description="A FastAPI service for interacting with OpenAI's GPT models.",
    version="1.0.0",
    docs_url="/gpt/docs",
    redoc_url="/gpt/redoc",
    openapi_url="/gpt/openapi.json"
)

# All routes included here will be prefixed with /gpt
app.include_router(lulu_router, prefix="/gpt")
app.include_router(myomyo_router, prefix="/gpt")


@app.get("/health", summary="Health Check", description="Check if the service is running.")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)

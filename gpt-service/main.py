import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import os

# It's good practice to load the API key from environment variables
# The user will need to set this in their secrets for the deploy.yml
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="GPT Service",
    description="A FastAPI service for interacting with OpenAI's GPT models.",
    version="1.0.0",
    docs_url="/gpt/docs",
    redoc_url="/gpt/redoc",
    openapi_url="/gpt/openapi.json"
)

class GptRequest(BaseModel):
    prompt: str = Field(description="The prompt to send to the GPT model.")
    model: str = Field(default="gpt-3.5-turbo", description="The model to use for the completion.")

class GptResponse(BaseModel):
    response: str = Field(description="The response from the GPT model.")

@app.post(
    "/gpt/exec",
    response_model=GptResponse,
    summary="Get a completion from a GPT model",
    description="Sends a prompt to the specified GPT model and returns the completion.",
    tags=["GPT"],
    responses={
        500: {"description": "Error interacting with OpenAI API."}
    }
)
async def get_gpt_completion(body: GptRequest):
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": body.prompt,
                }
            ],
            model=body.model,
        )
        response_content = chat_completion.choices[0].message.content
        if response_content is None:
            raise HTTPException(status_code=500, detail="Received an empty response from OpenAI.")
        return GptResponse(response=response_content)
    except Exception as e:
        # It's better to not expose the raw error message from the API in production
        # but for this context, it can be helpful for debugging.
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

@app.get("/health", summary="Health Check", description="Check if the service is running.")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    # Using a different port to avoid conflicts with other services
    uvicorn.run(app, host="0.0.0.0", port=8004)
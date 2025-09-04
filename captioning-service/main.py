from http.client import HTTPException

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import torch

app = FastAPI(
    title="Image Captioning Service",
    description="A FastAPI service for generating image captions using a pre-trained BLIP model.",
    version="1.0.0"
)

class ImageReq(BaseModel):
    image_url: str = Field(description="Image URL")

# print("모델 로드 중...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
captioning_model.to(device)
# print("모델 로드 완료.")


def preproc(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image

def generate_caption(image: Image.Image) -> str:
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = captioning_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

@app.post("/caption", summary="Generate image caption", description="Generate a caption for the given image URL.")
async def caption_image(req: ImageReq):
    try:
        response = requests.get(req.image_url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {e}")

    try:
        img = preproc(response.content)
        caption = generate_caption(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    return {"caption": caption }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

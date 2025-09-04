import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import easyocr
import numpy as np

app=FastAPI(
    title="Image Masking Service",
    description="A FastAPI service for masking images using a pre-trained segmentation model.",
    version="1.0.0"
)

class ImageReq(BaseModel):
    image_url: str

reader = easyocr.Reader(['en', 'ko'])

def mask_text(image: Image.Image) -> Image.Image:
    image_np = np.array(image)
    results = reader.readtext(image_np)

    text_threshold = 0.3

    filtered_boxes = [box for box, text, conf in results if conf >= text_threshold]

    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    for box in filtered_boxes:
        polygon = [(int(point[0]), int(point[1])) for point in box]
        draw.polygon(polygon, fill=(255, 255, 255))

    return masked

@app.post("/mask", summary="Mask text in image", description="Mask text in the given image URL.")
async def mask_image(req: ImageReq):
    try:
        response = requests.get(req.image_url)
        response.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"Failed to fetch image from URL: {e}"}

    # try:
    img = Image.open(BytesIO(response.content)).convert("RGB")
    masked_img = mask_text(img)
    #     buffered = BytesIO()
    #     masked_img.save(buffered, format="PNG")
    #     img_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()
    # except Exception as e:
    #     return {"error": f"Error processing image: {e}"}

    return {"masked_image": "masking completed"}
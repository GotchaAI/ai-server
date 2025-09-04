import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
from PIL import Image, ImageDraw
import easyocr
import numpy as np
import boto3
import os
import uuid

app=FastAPI(
    title="Image Masking Service",
    description="A FastAPI service for masking images using a pre-trained segmentation model.",
    version="1.0.0",
    docs_url="/mask/docs",
    redoc_url="/mask/redoc",
    openapi_url="/mask/openapi.json"
)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

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

@app.post("/mask/exec", summary="Mask text in image and upload to S3", description="Upload an image, mask text, and upload the masked image to AWS S3.")
async def mask_image(file: UploadFile = File(...)):
    if not S3_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="S3_BUCKET_NAME environment variable not set.")

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image file: {e}")

    masked_img = mask_text(img)

    # Save masked image to a BytesIO object
    masked_img_buffer = BytesIO()
    masked_img.save(masked_img_buffer, format="PNG")
    masked_img_buffer.seek(0) # Rewind the buffer to the beginning

    # Generate a unique filename for S3
    file_extension = file.filename.split(".")[-1] if "." in file.filename else "png"
    s3_filename = f"masked_images/{uuid.uuid4()}.{file_extension}"

    try:
        s3_client.upload_fileobj(masked_img_buffer, S3_BUCKET_NAME, s3_filename)
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_filename}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload masked image to S3: {e}")

    return {"message": "Masking and upload completed successfully", "s3_url": s3_url}

@app.get("/health", summary="Health Check", description="Check if the service is running.")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List
import requests, httpx
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import efficientnet_b0

async def lifespan(app):
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0),
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=200),
    )
    yield
    await app.state.http.aclose()

app = FastAPI(
    title="Image Classification Service",
    description="A FastAPI service for classifying images using a pre-trained EfficientNet model.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/classify/docs",
    redoc_url="/classify/redoc",
    openapi_url="/classify/openapi.json"
)

class ImageReq(BaseModel):
    image_url: str = Field(description="Image URL")

class AiPrediction(BaseModel):
    predicted: str = Field(description="Predicted category in Korean")
    confidence: float = Field(description="Confidence score")

class ClassifyRes(BaseModel):
    filename: str
    result: List[AiPrediction] = Field(description="List of predictions")

CATEGORIES= [
    "항공모함",
    "비행기",
    "알람시계",
    "앰뷸런스",
    "천사",
    "사과",
    "팔",
    "도끼",
    "백팩",
    "바나나",
    "붕대",
    "야구공",
    "농구공",
    "야구배트",
    "욕조",
    "침대",
    "꿀벌",
    "자전거",
    "새",
    "생일케이크",
    "책",
    "나비넥타이",
    "빵",
    "빗자루",
    "양동이",
    "버스",
    "수풀",
    "나비",
    "케이크",
    "달력",
    "카메라",
    "모닥불",
    "양초",
    "차",
    "당근",
    "고양이",
    "핸드폰",
    "의자",
    "교회",
    "동그라미",
    "구름",
    "컴파스",
    "컴퓨터",
    "쿠키",
    "소파",
    "소",
    "게",
    "악어",
    "왕관",
    "컵",
    "개",
    "돌고래",
    "도넛",
    "문",
    "오리",
    "귀",
    "코끼리",
    "편지봉투",
    "눈",
    "안경",
    "얼굴",
    "선풍기",
    "소화기",
    "물고기",
    "꽃",
    "포크",
    "개구리",
    "프라이팬",
    "정원",
    "기린",
    "포도",
    "기타",
    "망치",
    "모자",
    "헬리콥터",
    "육각형",
    "하키 채",
    "말",
    "아이스크림",
    "재킷",
            "캥거루", "키보드", "칼", "사다리", "노트북", "나뭇잎", "다리", "등대", "번개", "사자", "가재", "막대사탕", "우체통", "지도", "보드마카", "확성기", "달", "오토바이", "산", "머그컵"]

MODEL_PATH = "./models/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load efficientnet model
def load_model(model_path: str, num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

classifier = load_model(MODEL_PATH, num_classes=len(CATEGORIES))

# image preprocessing
def preproc(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image

encode_image = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def classify(image: Image.Image) -> List[dict]:
    img_tensor = encode_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = classifier(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top3_prob, top3_catid = torch.topk(probabilities, 3)
    results = []
    for i in range(top3_prob.size(0)):
        results.append({
            "predicted": CATEGORIES[top3_catid[i]],
            "confidence": top3_prob[i].item() * 100
        })
    return results

@app.post(
    "/classify/exec",
    response_model=ClassifyRes,
    summary="Classify image",
    description="Classify the given image URL.",
    tags=["Image Classification"],
    responses={
        415: {"description": "Unsupported content-type. Only images are allowed."},
        500: {"description": "Error processing image."}
    }
)
async def classify_image(request: Request, body: ImageReq):
    try:
        response = await request.app.http.state.get(body.image_url)
        response.raise_for_status()
        if not response.headers.get("content-type", "").startswith("image/"):
            raise HTTPException(415, "Unsupported content-type")
        img = preproc(response.content)
        predictions = classify(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    filename = body.image_url.split("/")[-1]
    result = [AiPrediction(**pred) for pred in predictions]
    return ClassifyRes(filename=filename, result=result)

@app.get("/health", summary="Health Check", description="Check if the service is running.")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
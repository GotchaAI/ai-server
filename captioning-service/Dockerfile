# ======================
# 1단계: Builder
# ======================
FROM python:3.9-slim AS builder

WORKDIR /app

# 빌드에 필요한 툴 설치 (예: gcc, g++, make)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치를 위한 requirements 복사
COPY requirements.txt .

# 가상환경(or 임시 디렉토리)에 패키지 설치
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ======================
# 2단계: Runtime
# ======================
FROM python:3.9-slim

WORKDIR /app

# builder 단계에서 설치한 라이브러리 복사
COPY --from=builder /install /usr/local

# 앱 소스 코드 복사
COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

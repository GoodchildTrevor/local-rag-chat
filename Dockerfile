# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Системные зависимости (для OCR, если нужно обрабатывать PDF/изображения)
RUN apt update && apt install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Installing deps with cashing
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy code (except .dockerignore)
COPY . .

# Copy entrypoint and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run entrypoint
ENTRYPOINT ["/entrypoint.sh"]
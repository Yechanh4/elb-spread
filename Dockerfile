FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ELB_DB=/data/market.sqlite
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY core ./core
COPY data/__init__.py data/providers.py data/cache.py data/service.py data/scheduler.py data/cli.py ./data/
COPY app ./app
COPY web ./web
RUN mkdir -p /data
EXPOSE 8780
# PORT is set by Render/Fly; 8780 locally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8780}"]

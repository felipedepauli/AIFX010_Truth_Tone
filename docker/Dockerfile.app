FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml README.md /app/
RUN uv pip install -e . --system
COPY src/ /app/src/
COPY frontend/ /app/frontend/
ENV APP_HOST=0.0.0.0 APP_PORT=8000
EXPOSE 8000
CMD ["uvicorn","dspfusion.service.api:app","--host","0.0.0.0","--port","8000","--proxy-headers","--root-path",""]

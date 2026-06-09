FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so they're cached when only app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local source (do not git clone at build time).
COPY . .

EXPOSE 8002

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "01_🧩_Playground.py", "--server.port=8002", "--server.address=0.0.0.0"]

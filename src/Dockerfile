FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY prompts.py .
COPY agent_build.py .
COPY streamlit_ui.py .
COPY fos_50_summary_records.csv .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
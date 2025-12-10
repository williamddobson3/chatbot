# Qwen Chatbot

A modern, production-ready chatbot application using Qwen2.5-72B-Instruct model.

## Features

- 🤖 Powered by Qwen2.5-72B-Instruct (72B parameters)
- 💬 Modern, responsive chat interface
- 🚀 FastAPI backend with async support
- 📦 Easy deployment to VPS
- 🔒 Secure and scalable architecture

## Prerequisites

- Python 3.10+
- ~145GB disk space for model
- CUDA-capable GPU (recommended) or sufficient RAM for CPU inference
- Git LFS (if downloading via git)

## Installation

### 1. Clone and Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model

The model will be automatically downloaded on first run, or you can pre-download it:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    local_dir="./models/Qwen2.5-72B-Instruct"
)
```

### 3. Configure Environment

Create a `.env` file:

```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Local Development

```bash
# Start the server
python main.py

# Or with uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access the chatbot at: `http://localhost:8000`

### Production Deployment

```bash
# Using gunicorn with uvicorn workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Or use the provided Docker setup:

```bash
docker-compose up -d
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── model.py         # Model loading and inference
│   └── config.py        # Configuration management
├── static/
│   ├── index.html       # Frontend interface
│   ├── style.css        # Styling
│   └── script.js        # Frontend logic
├── models/              # Model storage (gitignored)
├── .env                 # Environment variables
├── .env.example         # Example env file
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
└── README.md
```

## Configuration

Key environment variables:

- `MODEL_PATH`: Path to model (default: "Qwen/Qwen2.5-72B-Instruct")
- `DEVICE`: "cuda" or "cpu"
- `MAX_LENGTH`: Maximum generation length
- `TEMPERATURE`: Sampling temperature
- `TOP_P`: Top-p sampling parameter

## API Endpoints

- `GET /`: Chat interface
- `POST /api/chat`: Send message and get response
- `GET /api/health`: Health check

## License

This project uses the Qwen model which is licensed under the Qwen License.


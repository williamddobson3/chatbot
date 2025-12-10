#!/bin/bash

# Deployment script for VPS

set -e

echo "🚀 Deploying Qwen Chatbot..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration"
fi

# Create models directory if it doesn't exist
mkdir -p models

echo "✅ Deployment setup complete!"
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Or use gunicorn for production:"
echo "  gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000"


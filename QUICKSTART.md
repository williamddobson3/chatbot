# Quick Start Guide

## Local Development Setup

### 1. Initial Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp env.example .env

# Edit .env if needed (defaults work for most cases)
# For CPU-only: Set DEVICE=cpu
```

### 3. Download Model (Optional - will auto-download on first run)

```bash
# Option 1: Use the download script
python download_model.py

# Option 2: Manual download with git
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-72B-Instruct ./models/Qwen2.5-72B-Instruct
```

### 4. Run the Application

```bash
# Start the server
python main.py

# Or with uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the Chatbot

Open your browser and go to: `http://localhost:8000`

## VPS Deployment

### 1. Transfer Files to VPS

```bash
# Using SCP
scp -r . user@your-vps-ip:/path/to/destination

# Or use git
git clone <your-repo> /path/on/vps
```

### 2. Run Deployment Script

```bash
cd /path/to/project
chmod +x deploy.sh
./deploy.sh
```

### 3. Configure Environment

```bash
# Edit .env file
nano .env

# Important settings for VPS:
# - DEVICE=cuda (if GPU available) or DEVICE=cpu
# - HOST=0.0.0.0 (to accept external connections)
# - PORT=8000 (or your preferred port)
```

### 4. Start with Systemd (Production)

Create `/etc/systemd/system/qwen-chatbot.service`:

```ini
[Unit]
Description=Qwen Chatbot Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/project
Environment="PATH=/path/to/project/venv/bin"
ExecStart=/path/to/project/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable qwen-chatbot
sudo systemctl start qwen-chatbot
sudo systemctl status qwen-chatbot
```

### 5. Using Gunicorn (Alternative)

```bash
source venv/bin/activate
pip install gunicorn

# Start with gunicorn
gunicorn app.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300
```

### 6. Using Nginx as Reverse Proxy (Recommended)

Create `/etc/nginx/sites-available/qwen-chatbot`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/qwen-chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Docker Deployment

### 1. Build and Run

```bash
# Build image
docker-compose build

# Start container
docker-compose up -d

# View logs
docker-compose logs -f
```

### 2. Stop Container

```bash
docker-compose down
```

## Troubleshooting

### Model Not Loading

- Check disk space: `df -h`
- Verify model path in `.env`
- Check logs for download errors
- Ensure Hugging Face token is set if needed

### Out of Memory

- Use `DEVICE=cpu` if no GPU
- Reduce `MAX_LENGTH` in `.env`
- Consider using a smaller model variant

### Port Already in Use

- Change `PORT` in `.env`
- Or kill the process: `lsof -ti:8000 | xargs kill`

### Slow Responses

- Ensure GPU is being used: Check `nvidia-smi`
- Reduce `MAX_LENGTH` for faster generation
- Use quantization if supported

## Performance Tips

1. **GPU Memory**: Use `torch.float16` (default) to save memory
2. **Batch Size**: Currently set to 1, increase if you have more GPU memory
3. **Model Quantization**: Consider using 8-bit or 4-bit quantization for lower memory usage
4. **Caching**: Model weights are cached, first load is slower

## Security Notes

- Don't expose the API publicly without authentication
- Use HTTPS in production
- Consider rate limiting
- Validate and sanitize user inputs


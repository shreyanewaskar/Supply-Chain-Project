# Docker Deployment Guide

## Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed (included with Docker Desktop)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Build and run the container:**
```bash
docker-compose up -d
```

2. **Access the dashboard:**
Open browser: `http://localhost:8501`

3. **View logs:**
```bash
docker-compose logs -f
```

4. **Stop the container:**
```bash
docker-compose down
```

---

### Option 2: Using Docker Commands

1. **Build the image:**
```bash
docker build -t supply-chain-dashboard .
```

2. **Run the container:**
```bash
docker run -d \
  --name supply-chain-dashboard \
  -p 8501:8501 \
  -v $(pwd)/model:/app/model \
  supply-chain-dashboard
```

3. **Access the dashboard:**
Open browser: `http://localhost:8501`

4. **Stop the container:**
```bash
docker stop supply-chain-dashboard
docker rm supply-chain-dashboard
```

---

## Model Files Setup

### If models are NOT in the image:

**Option A: Mount local model directory**
```bash
docker run -d \
  -p 8501:8501 \
  -v /path/to/your/models:/app/model \
  supply-chain-dashboard
```

**Option B: Copy models into running container**
```bash
docker cp model/. supply-chain-dashboard:/app/model/
```

**Option C: Download models on startup**
Add to `dashboard.py`:
```python
import gdown

def download_models():
    # Download from Google Drive
    gdown.download('YOUR_GOOGLE_DRIVE_LINK', 'model/best.pt', quiet=False)
```

---

## Cloud Deployment

### Deploy to AWS ECR + ECS

1. **Build and tag:**
```bash
docker build -t supply-chain-dashboard .
docker tag supply-chain-dashboard:latest YOUR_AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/supply-chain-dashboard:latest
```

2. **Push to ECR:**
```bash
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker push YOUR_AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/supply-chain-dashboard:latest
```

3. **Deploy to ECS:**
- Create ECS cluster
- Create task definition using the ECR image
- Create service with load balancer

---

### Deploy to Google Cloud Run

1. **Build and push:**
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/supply-chain-dashboard
```

2. **Deploy:**
```bash
gcloud run deploy supply-chain-dashboard \
  --image gcr.io/YOUR_PROJECT_ID/supply-chain-dashboard \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

---

### Deploy to Azure Container Instances

1. **Build and push to ACR:**
```bash
az acr build --registry YOUR_REGISTRY --image supply-chain-dashboard .
```

2. **Deploy:**
```bash
az container create \
  --resource-group YOUR_RESOURCE_GROUP \
  --name supply-chain-dashboard \
  --image YOUR_REGISTRY.azurecr.io/supply-chain-dashboard \
  --dns-name-label supply-chain-dashboard \
  --ports 8501
```

---

### Deploy to DigitalOcean App Platform

1. **Push to Docker Hub:**
```bash
docker tag supply-chain-dashboard YOUR_DOCKERHUB_USERNAME/supply-chain-dashboard
docker push YOUR_DOCKERHUB_USERNAME/supply-chain-dashboard
```

2. **Create app on DigitalOcean:**
- Go to App Platform
- Select Docker Hub as source
- Enter image name
- Set port to 8501

---

## Environment Variables

Create `.env` file for configuration:
```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
MODEL_PATH=/app/model
```

Use with docker-compose:
```yaml
env_file:
  - .env
```

---

## Troubleshooting

### Container won't start
```bash
docker logs supply-chain-dashboard
```

### Out of memory
Increase memory limit:
```bash
docker run -d -p 8501:8501 --memory="4g" supply-chain-dashboard
```

### Models not loading
Check if models are in the correct directory:
```bash
docker exec supply-chain-dashboard ls -la /app/model
```

### Port already in use
Change port mapping:
```bash
docker run -d -p 8080:8501 supply-chain-dashboard
```
Access at: `http://localhost:8080`

---

## Production Recommendations

1. **Use multi-stage builds** to reduce image size
2. **Store models in cloud storage** (S3, GCS, Azure Blob)
3. **Use secrets management** for sensitive data
4. **Enable HTTPS** with reverse proxy (nginx, Traefik)
5. **Set up monitoring** (Prometheus, Grafana)
6. **Configure auto-scaling** based on CPU/memory
7. **Use CDN** for static assets
8. **Implement rate limiting** to prevent abuse

---

## Security Best Practices

1. **Don't run as root:**
Add to Dockerfile:
```dockerfile
RUN useradd -m -u 1000 streamlit
USER streamlit
```

2. **Scan for vulnerabilities:**
```bash
docker scan supply-chain-dashboard
```

3. **Use specific base image versions:**
```dockerfile
FROM python:3.9.18-slim
```

4. **Keep dependencies updated:**
```bash
pip list --outdated
```

---

## Performance Optimization

1. **Use .dockerignore** to exclude unnecessary files
2. **Layer caching** - put frequently changing files last
3. **Multi-stage builds** for smaller images
4. **Compress models** before including in image
5. **Use gunicorn** for production WSGI server

---

## Monitoring

### Check container stats:
```bash
docker stats supply-chain-dashboard
```

### View resource usage:
```bash
docker inspect supply-chain-dashboard
```

### Set up health checks:
Already configured in docker-compose.yml

---

## Backup and Recovery

### Backup models:
```bash
docker cp supply-chain-dashboard:/app/model ./model_backup
```

### Restore models:
```bash
docker cp ./model_backup/. supply-chain-dashboard:/app/model/
```

---

## Scaling

### Run multiple instances:
```bash
docker-compose up --scale streamlit-app=3
```

### Use load balancer (nginx):
```nginx
upstream streamlit {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}
```

---

## Support

For issues or questions:
1. Check logs: `docker logs supply-chain-dashboard`
2. Verify models are loaded
3. Check system resources
4. Review Streamlit documentation

---

## Cleanup

### Remove container:
```bash
docker-compose down
```

### Remove image:
```bash
docker rmi supply-chain-dashboard
```

### Remove all unused Docker resources:
```bash
docker system prune -a
```

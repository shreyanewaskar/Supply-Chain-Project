# Hugging Face Spaces Deployment Guide

## File Structure for Hugging Face Spaces

```
Supply_chain_management/
├── app.py                      # Main entry point (root level)
├── requirements.txt            # Python dependencies
├── packages.txt               # System dependencies
├── README.md                  # Hugging Face Space description
├── .gitattributes            # Git LFS configuration
├── app/
│   └── app.py                # Original dashboard code
├── model/                    # Model files (use Git LFS)
│   ├── half_model_25_Cinema_Media.h5
│   ├── half_model_25_Games_Software.h5
│   ├── best.pt
│   └── distilbert_model/
└── dataset/                  # Optional datasets
```

## Step-by-Step Deployment

### 1. Prepare Your Repository

**Option A: Create New Space on Hugging Face**

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in details:
   - **Space name**: supply-chain-ai-dashboard
   - **License**: MIT
   - **SDK**: Streamlit
   - **Hardware**: CPU basic (free) or upgrade for better performance

**Option B: Use Git to Push**

```bash
# Install Git LFS
git lfs install

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/supply-chain-ai-dashboard
cd supply-chain-ai-dashboard
```

---

### 2. Restructure Files for Hugging Face

**Create root-level `app.py`:**

```python
import sys
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent / 'app'
sys.path.insert(0, str(app_dir))

# Import the main dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import torch
import os
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Copy all code from app/app.py here
# OR import it as a module
```

**OR simpler approach - move `app/app.py` to root:**

```bash
# Copy app.py to root
cp app/app.py ./app.py

# Update model paths in app.py
# Change: ROOT_DIR = BASE_DIR.parent
# To: ROOT_DIR = BASE_DIR
```

---

### 3. Create Required Files

**Create `README.md` in root:**

```markdown
---
title: Supply Chain AI Dashboard
emoji: 📦
colorFrom: blue
colorTo: orange
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# Supply Chain AI Dashboard

[Your description here]
```

**Create `packages.txt`:**

```
libgl1-mesa-glx
libglib2.0-0
```

**Create `.gitattributes` for large files:**

```
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
```

---

### 4. Handle Large Model Files

**Option A: Use Git LFS (Recommended)**

```bash
# Track large files
git lfs track "*.h5"
git lfs track "*.pt"
git lfs track "model/**"

# Add and commit
git add .gitattributes
git add model/
git commit -m "Add models with Git LFS"
git push
```

**Option B: Download Models on Startup**

Add to your `app.py`:

```python
import os
import gdown
from huggingface_hub import hf_hub_download

def download_models():
    model_dir = Path('model')
    model_dir.mkdir(exist_ok=True)
    
    # Check if models exist
    if not (model_dir / 'best.pt').exists():
        st.info("Downloading models... This may take a few minutes on first run.")
        
        # Option 1: From Google Drive
        gdown.download(
            'https://drive.google.com/uc?id=YOUR_FILE_ID',
            str(model_dir / 'best.pt'),
            quiet=False
        )
        
        # Option 2: From Hugging Face Hub
        hf_hub_download(
            repo_id="YOUR_USERNAME/supply-chain-models",
            filename="best.pt",
            local_dir=str(model_dir)
        )

# Call at startup
download_models()
```

**Option C: Use Hugging Face Model Hub**

Upload models separately:

```bash
# Create a model repository
huggingface-cli repo create supply-chain-models --type model

# Upload models
huggingface-cli upload supply-chain-models model/ .
```

Then download in app:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="YOUR_USERNAME/supply-chain-models",
    local_dir="model"
)
```

---

### 5. Update requirements.txt

Make sure it includes:

```txt
streamlit>=1.28.0
pandas
numpy
plotly
Pillow
torch
torchvision
ultralytics
transformers
tensorflow
keras
scikit-learn
```

---

### 6. Push to Hugging Face

```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment"

# Push to Hugging Face
git push
```

---

### 7. Configure Space Settings

On Hugging Face Space page:

1. **Settings** → **Hardware**
   - Free: CPU basic (2 vCPU, 16GB RAM)
   - Paid: CPU upgrade, GPU (for faster inference)

2. **Settings** → **Secrets** (if needed)
   - Add API keys or credentials

3. **Settings** → **Variables**
   - Set environment variables

---

## Optimizations for Hugging Face

### 1. Reduce Model Size

```python
# Quantize models
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### 2. Lazy Loading

```python
@st.cache_resource
def load_models():
    # Load only when needed
    return models
```

### 3. Add Loading Indicators

```python
with st.spinner("Loading models..."):
    models = load_models()
```

### 4. Handle Timeouts

```python
import time

def load_with_retry(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if i == max_retries - 1:
                raise
            time.sleep(2 ** i)
```

---

## Troubleshooting

### Space Won't Build

**Check logs:**
- Go to Space → Logs tab
- Look for errors in build process

**Common issues:**
- Missing dependencies in requirements.txt
- System packages not in packages.txt
- File paths incorrect

### Models Not Loading

**Solutions:**
1. Check file paths are correct
2. Verify Git LFS is working: `git lfs ls-files`
3. Check model files uploaded: `ls -lh model/`
4. Add debug logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Model directory contents: {os.listdir('model')}")
```

### Out of Memory

**Solutions:**
1. Upgrade to paid hardware
2. Reduce model size (quantization)
3. Load models on-demand
4. Clear cache regularly

### Slow Performance

**Solutions:**
1. Use GPU hardware (paid)
2. Optimize model inference
3. Cache predictions
4. Reduce batch sizes

---

## Alternative: Streamlit Community Cloud

If Hugging Face has limitations, try Streamlit Cloud:

```bash
# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/supply-chain-dashboard.git
git push -u origin main

# Deploy on share.streamlit.io
# Select your GitHub repo
# Set main file: app/app.py
```

---

## Cost Comparison

| Platform | Free Tier | Paid Options |
|----------|-----------|--------------|
| **Hugging Face** | 2 vCPU, 16GB RAM | CPU upgrade, GPU |
| **Streamlit Cloud** | 1GB RAM, 1 CPU | N/A (limited) |
| **Heroku** | 512MB RAM | Dyno upgrades |
| **AWS EC2** | 750 hrs/month (1 year) | Pay per use |

---

## Best Practices

1. **Use Git LFS** for files > 10MB
2. **Add .gitignore** for cache files
3. **Document model sources** in README
4. **Add usage examples** with screenshots
5. **Set up CI/CD** for automatic updates
6. **Monitor usage** and costs
7. **Add error handling** for model loading
8. **Implement caching** for better performance

---

## Example Spaces for Reference

- [Stable Diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion)
- [YOLO Object Detection](https://huggingface.co/spaces/Xenova/yolov8-object-detection)
- [Sentiment Analysis](https://huggingface.co/spaces/cardiffnlp/twitter-roberta-base-sentiment)

---

## Support

- [Hugging Face Docs](https://huggingface.co/docs/hub/spaces)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Git LFS Guide](https://git-lfs.github.com/)

---

## Quick Checklist

- [ ] Create Hugging Face Space
- [ ] Restructure files (app.py in root)
- [ ] Create README.md with metadata
- [ ] Add packages.txt for system deps
- [ ] Configure Git LFS for large files
- [ ] Upload model files
- [ ] Test locally first
- [ ] Push to Hugging Face
- [ ] Check build logs
- [ ] Test deployed app
- [ ] Share your Space! 🚀

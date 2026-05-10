# Supply Chain AI Dashboard

An AI-powered supply chain management dashboard built with Streamlit, combining demand forecasting, object detection, and sentiment analysis to optimize inventory and supply chain decisions.

---

## Features

- **Demand Forecasting** — Half-LSTM & BiLSTM models predict next 3 months of sales demand per shop and category
- **Object Detection** — YOLOv8 detects and counts warehouse inventory items from images
- **Sentiment Analysis** — Fine-tuned DistilBERT model analyzes supplier/customer feedback
- **Shortage Alerts** — Proactive inventory monitoring with 90-day stock projection
- **AI-Powered Insights** — Combines Half-LSTM forecasting with NLP sentiment scores for adjusted demand predictions

---

## Datasets

| Dataset | Source |
|---|---|
| Sales Forecasting | [Predict Future Sales — Kaggle](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales) |
| Sentiment Analysis | [Amazon Fashion 800K User Reviews — Kaggle](https://www.kaggle.com/datasets/fawadhossaini1415/amazon-fashion-800k-user-reviews-dataset) |

---

## Models

| Model | Purpose | Location |
|---|---|---|
| BiLSTM / Half-LSTM | Demand Forecasting | `model/*.h5` |
| YOLOv8 | Object Detection | `model/best.pt` |
| DistilBERT (Fine-tuned) | Sentiment Analysis | Google Drive (auto-downloaded) |

> The DistilBERT model is automatically downloaded from Google Drive on first run.

---

## Model Performance

### Demand Forecasting — Model Comparison

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| ARIMA | 0.277 ± 0.121 | 0.830 ± 0.844 | 29.9% ± 14.8% |
| Prophet (Tuned) | 0.288 ± 0.171 | 0.756 ± 0.709 | 27.0% ± 21.4% |
| XGBoost | 0.246 ± 0.000 | 0.072 ± 0.000 | 21.6% ± 0.0% |
| LightGBM | 0.241 ± 0.000 | 0.066 ± 0.000 | 21.1% ± 0.0% |
| Vanilla LSTM | 0.249 ± 0.000 | 0.073 ± 0.000 | 19.0% ± 0.0% |
| Transformer | 0.211 ± 0.000 | 0.062 ± 0.000 | 17.5% ± 0.0% |
| N-BEATS | 0.230 ± 0.000 | 0.072 ± 0.000 | 17.5% ± 0.0% |
| HALF-LSTM v4 (Ours) | 0.160 ± 0.131 | 0.397 ± 0.239 | 13.3% ± 10.9% |
| **BiLSTM v3 (Ours) ◄ BEST** | **0.154 ± 0.112** | **0.394 ± 0.265** | **12.7% ± 9.1%** |

### Sentiment Analysis — Model Comparison

| Model | Accuracy | Type |
|---|---|---|
| VADER (Baseline) | 66.91% | Rule-based |
| Logistic Regression | 87.15% | Machine Learning |
| **DistilBERT (Fine-tuned) ◄ BEST** | **89.83%** | Deep Learning |

---

## Project Structure

```
Supply_chain_management/
├── app/
│   └── dashboard.py        # Main Streamlit app
├── model/
│   ├── *.h5                # LSTM/BiLSTM models
│   ├── best.pt             # YOLOv8 model
│   └── distilbert_model/   # DistilBERT (auto-downloaded)
├── Notebook/
│   ├── supply_chain_noteboook.ipynb
│   ├── object_detection_final.ipynb
│   └── Sentiment_Analysis.ipynb
├── requirements.txt
└── Dockerfile
```

---

## Installation & Run

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd Supply_chain_management
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the dashboard**
```bash
streamlit run app/dashboard.py
```

Open your browser at `http://localhost:8501`

---

## Docker

```bash
docker-compose up --build
```

---

## Tech Stack

- **Frontend** — Streamlit
- **Forecasting** — TensorFlow, Keras (LHaSTM, BiLSTM)
- **Object Detection** — YOLOv8 (Ultralytics)
- **NLP** — HuggingFace Transformers (DistilBERT)
- **Visualization** — Plotly
- **Containerization** — Docker

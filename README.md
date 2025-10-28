# Customer Satisfaction Prediction Pipeline with ZenML

A comprehensive MLOps pipeline for predicting customer satisfaction scores using ZenML, MLflow, and Streamlit. This project demonstrates end-to-end machine learning workflows from data ingestion to model deployment and web interface.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Tutorial Reference](#tutorial-reference)


## 🎯 Overview

This project predicts customer satisfaction scores (0-5 scale) based on order and product features. It implements a complete MLOps pipeline using ZenML for orchestration, MLflow for experiment tracking and model deployment, and Streamlit for the web interface.

**Key Features:**
- Automated data preprocessing and cleaning
- Model training with multiple algorithms (Linear Regression, LightGBM, XGBoost)
- Model evaluation and deployment automation
- Real-time prediction API
- Interactive web interface
- Experiment tracking and model versioning

## 🚀 Features

### Data Processing
- **Data Ingestion**: Automated CSV data loading
- **Data Cleaning**: Missing value imputation and feature engineering
- **Feature Engineering**: 12 key features for prediction

### Model Training
- **Multiple Algorithms**: Linear Regression, LightGBM, XGBoost
- **Cross-validation**: Automated model evaluation
- **Hyperparameter Tuning**: Optimized model performance

### MLOps Pipeline
- **ZenML Orchestration**: Automated pipeline execution
- **MLflow Integration**: Experiment tracking and model registry
- **Model Deployment**: Automated deployment with quality gates
- **Monitoring**: Pipeline and model performance monitoring

### Web Interface
- **Streamlit App**: Interactive prediction interface
- **Real-time Predictions**: Instant customer satisfaction scoring
- **Feature Input**: User-friendly form for all 12 features

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  ZenML Pipeline │───▶│  Model Training │
│   (CSV Files)   │    │   Orchestration │    │   & Evaluation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  MLflow Tracking│◀───│  Model Registry │───▶│ Model Deployment│
│   & Registry    │    │   & Versioning  │    │   (MLflow)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Streamlit App  │
                       │  (Web Interface)│
                       └─────────────────┘
```

## 📋 Prerequisites

- Python 3.10+
- pip
- Git
- WSL2 (if running on Windows)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/lyly98/CUSTOMER_SATISFACTION.git
cd CUSTOMER_SATISFACTION
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install ZenML Integrations
```bash
zenml integration install mlflow -y
```

### 5. Initialize ZenML
```bash
zenml init
```

### 6. Register ZenML Components
```bash
# Create experiment tracker
zenml experiment-tracker register customer_satisfaction_tracker --flavor=mlflow

# Create model deployer
zenml model-deployer register customer_satisfaction_deployer --flavor=mlflow

# Create and activate stack
zenml stack register customer_satisfaction_stack \
  -a default \
  -o default \
  -d customer_satisfaction_deployer \
  -e customer_satisfaction_tracker \
  --set
```

## 🚀 Usage

### 1. Train and Deploy Model
```bash
python run_deployement.py --config deploy
```

### 2. Run Predictions
```bash
python run_deployement.py --config predict
```

### 3. Launch Web Interface
```bash
streamlit run streamlit_app.py
```

The Streamlit app will be available at `http://localhost:8501`

### 4. View MLflow Dashboard
```bash
mlflow ui --backend-store-uri 'file:/home/ai_sodira/.config/zenml/local_stores/[your-store-id]/mlruns'
```

## 📁 Project Structure

```
CUSTOMER_SATISFACTION/
├── data/
│   └── olist_customers_dataset.csv    # Training data
├── src/
│   └── data_cleaning.py               # Data preprocessing
├── steps/
│   ├── ingest_data.py                 # Data ingestion step
│   ├── clean_data.py                  # Data cleaning step
│   ├── model_train.py                 # Model training step
│   └── evaluation.py                  # Model evaluation step
├── piplines/
│   ├── deployment_pipeline.py         # Main deployment pipeline
│   ├── training_pipeline.py           # Training pipeline
│   └── utils.py                       # Utility functions
├── run_deployement.py                 # Main execution script
├── run_pipeline.py                    # Pipeline runner
├── streamlit_app.py                   # Web interface
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```


### Feature Importance
The model considers the following 12 features for prediction:

1. **Payment Sequential** - Payment method sequence
2. **Payment Installments** - Number of installments
3. **Payment Value** - Total payment amount
4. **Price** - Product price
5. **Freight Value** - Shipping cost
6. **Product Name Length** - Length of product name
7. **Product Description Length** - Length of product description
8. **Product Photos Quantity** - Number of product photos
9. **Product Weight (g)** - Product weight in grams
10. **Product Length (cm)** - Product length in centimeters
11. **Product Height (cm)** - Product height in centimeters
12. **Product Width (cm)** - Product width in centimeters

## 🎓 Tutorial Reference

This project was developed following the comprehensive tutorial: **[ZenML Customer Satisfaction Pipeline Tutorial](https://www.youtube.com/watch?v=-dJPoLm_gtE&t=10690s)**

The tutorial covers:
- ZenML pipeline setup and configuration
- MLflow integration for experiment tracking
- Model deployment strategies
- Streamlit web interface development
- MLOps best practices

**Key learnings from the tutorial:**
- ZenML pipeline orchestration
- MLflow model registry and deployment
- Automated model evaluation and deployment triggers
- Web interface development with Streamlit
- MLOps pipeline design patterns


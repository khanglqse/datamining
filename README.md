# Business Health Analysis and Fraud Detection Project

This project implements a data mining pipeline to analyze business health and detect potential fraud using data from cafef.vn.

## Project Structure

```
.
├── data/                   # Directory for storing raw and processed data
├── src/                    # Source code
│   ├── crawler/           # Data crawling modules
│   ├── preprocessing/     # Data preprocessing modules
│   └── modeling/         # Modeling and analysis modules
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Features

1. **Data Collection**
   - Crawls financial data from cafef.vn
   - Collects company information, financial statements, and news

2. **Data Preprocessing**
   - Data cleaning and normalization
   - Feature engineering
   - Missing value handling

3. **Modeling**
   - Business health assessment
   - Fraud detection using machine learning models
   - Performance evaluation and visualization

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the crawler to collect data:
```bash
python src/crawler/cafef_crawler.py
```

2. Preprocess the collected data:
```bash
python src/preprocessing/preprocess.py
```

3. Run the modeling pipeline:
```bash
python src/modeling/train.py
```

## Data Sources

The project collects data from:
- Company financial statements
- News articles
- Market data
- Company profiles

## Models

The project implements several models for:
- Business health assessment
- Fraud detection
- Anomaly detection

## License

MIT License 
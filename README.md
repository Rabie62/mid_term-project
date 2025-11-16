
## Problem Description

**Goal**: Binary classification — predict if a patient has heart disease (`1`) or not (`0`).  
**Dataset**: UCI Heart Disease (Cleveland processed) – 303 patients, 14 clinical features.  
**Use Case**: Assist doctors in early screening and risk assessment.

## Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **File**: `data/heart.csv` (included in repo)
- **Features**:
  - `age`, `sex`, `cp` (chest pain), `trestbps` (resting BP), `chol` (cholesterol), etc.
  - `target`: 1 = heart disease, 0 = no disease

## How to Run

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt

# financial-feature-selection
Feature selection for bankruptcy prediction on an American Companies Bankruptcy dataset.

## Methods
- **Tabu Search** with kNN or Random Forest as the evaluation function
- **RFE** (Recursive Feature Elimination) with Random Forest via cross-validated RFECV

## Usage
Set `METHOD` in `main.py` to one of: `"tabu+knn"`, `"tabu+rf"`, or `"rfe"`, then run:
```
python main.py
```

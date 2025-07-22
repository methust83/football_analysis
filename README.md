A Machine Learning model to predict football match outcomes (Win, Draw, Lose) using historical data, team statistics, and player performance metrics. Built with Python 🐍 and scikit-learn.

---

## 📖 Features

- 🏟 Predict match results: Home Win, Draw, Away Win
- 📊 Uses team & player statistics, ELO ratings, recent form
- 🔥 Models: Logistic Regression, Random Forest, XGBoost
- 🌐 API ready (Flask/Django integration)
- 📈 Visualization of predictions and confidence scores

---

## 🚀 Demo

| Home Team | Away Team | Prediction | Confidence |
|-----------|-----------|-------------|------------|
| Chelsea    | Arsenal   | Home Win     | 72%        |
| Man Utd    | Liverpool | Draw         | 54%        |
| Barcelona  | Real Madrid| Away Win    | 68%        |

---

## 🛠 Installation

1. Clone the repository
    ```bash
    git clone https://github.com/methust83/football_analysis.git
    cd football_analysis
    ```

2. Create virtual environment & install dependencies
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt

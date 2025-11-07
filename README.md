# SPARX - Prototype (Sports Performance & Injury Risk Analytics)

This is a simple software prototype for the SPARX project — an AI-based sports performance monitoring and injury risk prediction system.

## What's included
- `app.py` — Streamlit app to test the model (load `model.joblib` and accept CSV/manual input)
- `train_model.py` — regenerate synthetic dataset and the model (`model.joblib`)
- `model.joblib` — trained logistic regression model
- `sample_data.csv` — synthetic sample dataset
- `requirements.txt` — Python dependencies

## How to run locally
1. (Optional) Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Or regenerate model & data:
   ```bash
   python train_model.py
   ```

## How to prepare GitHub submission
1. Create a new GitHub repository named `SPARX-prototype`.
2. Copy all files and push:
   ```bash
   git init
   git add .
   git commit -m "Initial SPARX prototype"
   git branch -M main
   git remote add origin <your-github-https-or-ssh-url>
   git push -u origin main
   ```
3. Use the repository link as the submission URL in the hackathon form.

## Notes
- This prototype uses a synthetic dataset and a simple logistic regression. For real deployment, collect real athlete training logs and iteratively improve model performance.
- Suggested next steps: add explainability, dashboard visualizations, and user auth for multiple athletes.

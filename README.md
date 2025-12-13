# Road-Accident-Risk-Prediction

This repository contains a minimal Streamlit app to predict road accident risk using a regression model.

How to provide a trained model
- Place a trained scikit-learn regressor saved with `joblib.dump(model, 'model.pkl')` in the repository root. The app will automatically load `model.pkl` if present.
- Alternatively, when the Streamlit app is running you can upload a `.pkl` or `.joblib` file using the "Upload trained model" control — the uploaded file will be saved as `model.pkl` and used for predictions.

Run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

If you'd like, I can add a training script that trains a model from a CSV dataset and saves `model.pkl`. Provide the dataset or tell me its path/format and I'll add it.
# Road-Accident-Risk-Prediction
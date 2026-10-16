# AnemiaScan - Anemia Prediction Web App (Flask + ML + REST API)

Enter Gender, Hemoglobin, MCH, MCHC and MCV from a CBC report and the
Random Forest model tells you whether the values point to anemia.

## Project structure
```
app.py                 Flask app + REST API (this is what Vercel runs)
train_model.py         Trains the model (creates model.joblib + metrics.json)
model.joblib           Trained Random Forest
metrics.json           Accuracy, precision, recall, feature importance
data/anemia.csv        Dataset
templates/index.html   Frontend (HTML + CSS + JS in one file)
requirements.txt       Dependencies
```

## Run locally
```
python -m venv venv
venv\Scripts\activate          # Windows (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py                  # open http://127.0.0.1:5000
```
To retrain: `python train_model.py`

## API
| Method | Route | Description |
|---|---|---|
| GET | /api/health | Server status |
| GET | /api/model-info | Model accuracy, feature importance |
| POST | /api/predict | Make a prediction |

Example request body for POST /api/predict:
```
{"gender": 0, "Hemoglobin": 9.5, "MCH": 20, "MCHC": 29, "MCV": 72}
```
gender: 0 = Female, 1 = Male.

## Deploy on Vercel
1. Push this folder to a GitHub repository.
2. On vercel.com choose Add New > Project and import the repo.
3. Keep the default settings (Flask is detected automatically) and click Deploy.

Or with the CLI: `npm i -g vercel`, then `vercel --prod` inside this folder.

Disclaimer: educational screening tool, not a medical diagnosis.

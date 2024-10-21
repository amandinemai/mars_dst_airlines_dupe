import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pymongo import MongoClient
from .fastapi_users import users
from .fastapi_models import Flight
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import mlflow
import logging
from typing import List

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Définition des constantes
security = HTTPBasic()
app = FastAPI()

# Connexion à la base de données
try:
    mongo_uri = os.getenv("MONGO_URI", "mongodb://admin:password@localhost:27017/")
    client = MongoClient(mongo_uri)
    db = client["lufthansa"]
    collection = db["Status"]
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    raise HTTPException(status_code=500, detail="Database connection failed")

# Fonction d'authentification
def authenticate(creds: HTTPBasicCredentials = Depends(security)):
    username = creds.username
    password = creds.password
    if users.get(username) == password:
        return True
    raise HTTPException(status_code=401, detail="Authentication failed")

@app.get("/flightInformation")
async def get_flight(flightNumber: str, Authorization: bool = Depends(authenticate)):
        projection = {"MarketingCarrier.FlightNumber": 1, "Departure.AirportCode": 1, 
                      "Departure.ScheduledTimeUTC": 1, "Departure.TimeStatus.Definition": 1, 
                      "Departure.Terminal.Gate": 1, "Arrival.AirportCode": 1, 
                      "Arrival.ScheduledTimeUTC": 1, "Arrival.TimeStatus.Definition": 1, "_id": 0}
        filtered_flights = collection.find_one({"MarketingCarrier.FlightNumber": flightNumber}, projection)
        if filtered_flights is None:
            # Si aucun vol n'est trouvé, retourner une réponse 404
            raise HTTPException(status_code=404, detail="Flight not found")
        return filtered_flights


# Fonction de récupération de l'expérience MLflow
def get_mlflow_experiment(experiment_name: str):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist")
    return experiment

# Fonction de prédiction de vol
@app.get("/flightPrediction")
async def get_prediction(flightNumber: str, Authorization: bool = Depends(authenticate)):
    try:
        tracking_uri = 'http://127.0.0.1:5000'
        mlflow.set_tracking_uri(tracking_uri)

        experiment_name = "lufthansa_flight_status_ML"
        experiment = get_mlflow_experiment(experiment_name)
        experiment_id = experiment.experiment_id
        client = MlflowClient()
        runs = client.search_runs(experiment_ids=[experiment_id])

        linear_svc_runs = [run for run in runs if run.data.params.get("model_type") == "LinearSVC"]
        if not linear_svc_runs:
            raise ValueError("No runs found for the LinearSVC model")

        linear_svc_runs.sort(key=lambda run: run.info.start_time, reverse=True)
        latest_run = linear_svc_runs[0]
        latest_run_id = latest_run.info.run_id

        model_uri = f"runs:/{latest_run_id}/model"
        print(model_uri)
        loaded_model = mlflow.sklearn.load_model(model_uri)

        data = collection.find_one({"MarketingCarrier.FlightNumber": flightNumber})
        if not data:
            raise HTTPException(status_code=404, detail="Flight data not found")

        df = pd.json_normalize(data)
        # Aplatir les champs imbriqués
        df = pd.json_normalize(data)

        df = df[['Departure.TimeStatus.Code', 'Arrival.TimeStatus.Code', 
                 'Departure.ScheduledTimeUTC.DateTime', 'Departure.ActualTimeUTC.DateTime', 
                 'Arrival.ScheduledTimeUTC.DateTime', 'Arrival.ActualTimeUTC.DateTime']]
        df = df.drop_duplicates()
        df = df.dropna()

        df['duree'] = (pd.to_datetime(df['Arrival.ScheduledTimeUTC.DateTime']) - 
                       pd.to_datetime(df['Departure.ScheduledTimeUTC.DateTime'])).dt.total_seconds() / 60
        df['retard'] = (pd.to_datetime(df['Departure.ActualTimeUTC.DateTime']) - 
                        pd.to_datetime(df['Departure.ScheduledTimeUTC.DateTime'])).dt.total_seconds() / 60

        df_features = df[['Departure.TimeStatus.Code', 'duree', 'retard']]

        mapping = {'FE': 0, 'OT': 0, 'DL': 1}
        df['Arrival.TimeStatus.mapped'] = df['Arrival.TimeStatus.Code'].map(mapping).astype(bool)

        df_features['Departure.TimeStatus.mapped'] = df_features['Departure.TimeStatus.Code'].map(mapping).astype(bool)
        df_features = df_features.drop(columns=['Departure.TimeStatus.Code'])
        le = LabelEncoder()
        df_features['Departure.TimeStatus.mapped'] = le.fit_transform(df_features['Departure.TimeStatus.mapped'])

        df_features = df_features[['Departure.TimeStatus.mapped', 'duree', 'retard']]

        prediction = loaded_model.predict(df_features)
        prediction_list = prediction.tolist()

        return {"prediction": prediction_list}
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/flightCollection", response_model=List[Flight])
async def get_flight_collection(Authorization: bool = Depends(authenticate)):
    projection = {"MarketingCarrier.FlightNumber": 1, "Departure.AirportCode": 1, 
                      "Departure.ScheduledTimeUTC": 1, "Departure.TimeStatus.Definition": 1, "Arrival.AirportCode": 1, 
                      "Arrival.ScheduledTimeUTC": 1, "Arrival.TimeStatus.Definition": 1, "_id": 0}
    delayed_flights = collection.find({},projection)
    if delayed_flights is None:
        # Si aucun vol n'est trouvé, retourner une réponse 404
        raise HTTPException(status_code=404, detail="Flight not found")
    return list(delayed_flights)

    
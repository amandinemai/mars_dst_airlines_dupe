import pandas as pd
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn

# Connect to MongoDB
client = MongoClient("mongodb://admin:password@localhost:27017/")
db = client["lufthansa"]
collection = db["StatusML"]

# Récupération des données de la collection
data = list(collection.find())

# Création du DataFrame
df = pd.DataFrame(data)

# Aplatir les champs imbriqués
df = pd.json_normalize(data)

# Sélection des colonnes
df = df[['Departure.TimeStatus.Code', 'Arrival.TimeStatus.Code', 
         'Departure.ScheduledTimeUTC.DateTime', 'Departure.ActualTimeUTC.DateTime', 
         'Arrival.ScheduledTimeUTC.DateTime', 'Arrival.ActualTimeUTC.DateTime']]

df = df.drop_duplicates()
df = df.dropna()

# Calcul de la durée du vol
df['duree'] = (pd.to_datetime(df['Arrival.ScheduledTimeUTC.DateTime']) - 
               pd.to_datetime(df['Departure.ScheduledTimeUTC.DateTime']))
df['duree'] = df['duree'].dt.total_seconds() / 60

# Calcul du retard du vol
df['retard'] = (pd.to_datetime(df['Departure.ActualTimeUTC.DateTime']) - 
                pd.to_datetime(df['Departure.ScheduledTimeUTC.DateTime']))
df['retard'] = df['retard'].dt.total_seconds() / 60

df_features = df[['Departure.TimeStatus.Code', 'duree', 'retard']]

# Mapping / Encodage
mapping = {'FE': 0, 'OT': 0, 'DL': 1}
df['Arrival.TimeStatus.mapped'] = df['Arrival.TimeStatus.Code'].map(mapping).astype(bool)
df_target = df['Arrival.TimeStatus.mapped']

df_features['Departure.TimeStatus.mapped'] = df_features['Departure.TimeStatus.Code'].map(mapping).astype(bool)
df_features = df_features.drop(columns=['Departure.TimeStatus.Code'])

# Encode target variable
le = LabelEncoder()
df_target = le.fit_transform(df_target)
df_features['Departure.TimeStatus.mapped'] = le.fit_transform(df_features['Departure.TimeStatus.mapped'])

df_features = df_features[['Departure.TimeStatus.mapped', 'duree', 'retard']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.30, random_state=42)

# Standardize numeric variables
scaler = StandardScaler()
X_train[['duree', 'retard']] = scaler.fit_transform(X_train[['duree', 'retard']])
X_test[['duree', 'retard']] = scaler.transform(X_test[['duree', 'retard']])

# LinearSVC model
model = LinearSVC(dual="auto")

mlflow.set_experiment("lufthansa_flight_status_ML")

# Start an MLflow run
with mlflow.start_run() as run:
    try:
        # Train the model
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log parameters
        mlflow.log_param("model_type", "LinearSVC")
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        
        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Model accuracy: {accuracy}")
        
        # Classification report
        print(classification_report(y_test, y_pred))
        
        # Classification report 
        classification_report_str = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(classification_report_str, "classification_report.json")
        
        # Plot feature importance
        coefficients = model.coef_[0]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(coefficients)), coefficients)
        plt.xticks(range(len(coefficients)), df_features.columns, rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Coefficient')
        plt.title('Importance des variables features dans le modèle LinearSVC')
        
        # Save plot
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        
        plt.show()
        
        run_id = run.info.run_id
        mlflow.log_param("run_id", run_id)
        print(f"Run ID: {run_id}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        mlflow.log_param("error", str(e))

import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# Connection a mangodb
client = MongoClient("mongodb://admin:password@localhost:27017/")
db = client["lufthansa"]
collection = db["StatusML"]

# Recuperation des champs importants 
projection = {
    "Departure.AirportCode": 1,
    "Departure.ScheduledTimeUTC": 1,
    "Departure.TimeStatus.Code": 1,
    "Departure.Terminal.Gate": 1,
    "Arrival.AirportCode": 1,
    "Arrival.ScheduledTimeUTC": 1,
    "Arrival.TimeStatus.Code": 1,
    "_id": 0
}

# Recuperation des données depuis monngodb et reconvertion en dataframe pandas afin de traiter les données
data = list(collection.find({}, projection))
df = pd.DataFrame(data)
df = pd.json_normalize(data)

# exclure les codes iata qui apparaissent une seule fois dans les airoport de depart  
departure_airportcode = df['Departure.AirportCode'].value_counts()
departure_airportcode_occur_once = departure_airportcode[departure_airportcode == 1].index.tolist()
print('Aéroports de départ qui apparaissent une seule fois à exclure: ', departure_airportcode_occur_once)
df = df.loc[~df['Departure.AirportCode'].isin(departure_airportcode_occur_once)]
# exclure les codes iata qui apparaissent une seule fois dans les airoport d'arrivee 
arrival_airportcode = df['Arrival.AirportCode'].value_counts()
arrival_airportcode_occur_once = arrival_airportcode[arrival_airportcode == 1].index.tolist()
print('Aéroports d arrivée qui apparaissent une seule fois à exclure: ', arrival_airportcode_occur_once)
df = df.loc[~df['Arrival.AirportCode'].isin(arrival_airportcode_occur_once)]

# Convertir en date et extraire le jour de la semaine
df['Departure.DateTime'] = pd.to_datetime(df['Departure.ScheduledTimeUTC.DateTime'])
df['Arrival.DateTime'] = pd.to_datetime(df['Arrival.ScheduledTimeUTC.DateTime'])

df['Departure.DayOfWeek'] = df['Departure.DateTime'].dt.day_of_week
df['Arrival.DayOfWeek'] = df['Arrival.DateTime'].dt.day_of_week

# Supprimer les colonnes inutiles 
df.drop([
    'Departure.Terminal.Gate',
    'Departure.ScheduledTimeUTC.DateTime',
    'Arrival.ScheduledTimeUTC.DateTime',
    'Departure.DateTime',
    'Arrival.DateTime'
], axis=1, inplace=True)

df = df.reindex(columns=[
    'Departure.AirportCode',
    'Departure.DayOfWeek',
    'Departure.TimeStatus.Code',
    'Arrival.AirportCode',
    'Arrival.DayOfWeek',
    'Arrival.TimeStatus.Code'
])

# Separate features and target
feats = df.drop('Arrival.TimeStatus.Code', axis=1)
target = df['Arrival.TimeStatus.Code']

# Séparer les caractéristiques et la cible
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.20, random_state=42)

# Encoder les variables catégorielles
cat = ['Departure.AirportCode', 'Departure.TimeStatus.Code', 'Arrival.AirportCode']

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999)
X_train.loc[:, cat] = oe.fit_transform(X_train[cat])
X_test.loc[:, cat] = oe.transform(X_test[cat])

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Decision Tree Classifier
dt = DecisionTreeClassifier()

mlflow.set_experiment("lufthansa_flight_status_ML")
# Start MLflow run
with mlflow.start_run():
    # Fit le model
    dt.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("model_type", "DecisionTreeClassifier")
    y_pred = dt.predict(X_test)
    # Log metrics
    train_score = dt.score(X_train, y_train)
    test_score = dt.score(X_test, y_test)
    mlflow.log_metric("train_score", train_score)
    mlflow.log_metric("test_score", test_score)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    print(f"Model accuracy: {accuracy}")
    print('Score sur ensemble train', train_score)
    print('Score sur ensemble test', test_score)

    # Evaluation du model
    y_pred_dt = dt.predict(X_test)

    print(classification_report(y_test, y_pred_dt))
    pd.crosstab(y_test, y_pred_dt, rownames=['Realité'], colnames=['Prédiction'])

    # Log model
    mlflow.sklearn.log_model(dt, "decision_tree_model")

    # Log les featurew importantes
    feat_importances = pd.DataFrame(dt.feature_importances_, index=feats.columns, columns=["Importance"])
    feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
    feat_importances.plot(kind='bar', figsize=(8, 6))
    plt.show()

    # Enregistrement de l'artifact : 
    plt.savefig("feature_importances.png")
    mlflow.log_artifact("feature_importances.png")
    run_id = mlflow.active_run().info.run_id
    mlflow.log_param("run_id", run_id)
    print(f"Run ID: {run_id}")
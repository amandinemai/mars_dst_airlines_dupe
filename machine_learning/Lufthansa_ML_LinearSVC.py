import pandas as pd
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Connexion à MongoDB
client = MongoClient("mongodb://admin:password@localhost:27017/")
db = client["lufthansa"]
collection = db["Status"]

# Récupération des données de la collection
data = list(collection.find())

# Création du DataFrame
df = pd.DataFrame(data)

# Aplatir les champs imbriqués
df = pd.json_normalize(data)

df = df[['Departure.TimeStatus.Code','Arrival.TimeStatus.Code','Departure.ScheduledTimeUTC.DateTime','Departure.ActualTimeUTC.DateTime','Arrival.ScheduledTimeUTC.DateTime','Arrival.ActualTimeUTC.DateTime']]

df = df.drop_duplicates()
df = df.dropna()

# CALCUL DE LA DUREE DU VOL 
df['duree']=(pd.to_datetime(df['Arrival.ScheduledTimeUTC.DateTime'])-pd.to_datetime(df['Departure.ScheduledTimeUTC.DateTime']))
df['duree']= df['duree'].dt.total_seconds()/60

# CALCUL DU RETARD DU VOL
df['retard']=(pd.to_datetime(df['Departure.ActualTimeUTC.DateTime'])-pd.to_datetime(df['Departure.ScheduledTimeUTC.DateTime']))
df['retard'] =df['retard'].dt.total_seconds()/60

df_features = df[['Departure.TimeStatus.Code','duree','retard']]

# MAPPING / ENCODAGE

# Créer un dictionnaire de remplacement
mapping = {'FE': 0, 'OT': 0, 'DL': 1}

# Appliquer le mapping à la colonne cible
df.loc[:, 'Arrival.TimeStatus.mapped'] = df['Arrival.TimeStatus.Code'].map(mapping).astype(bool)
df_target = df['Arrival.TimeStatus.mapped']

# Modification des colonne features 
df_features.loc[:, 'Departure.TimeStatus.mapped'] = df_features['Departure.TimeStatus.Code'].map(mapping).astype(bool)
del df_features['Departure.TimeStatus.Code']

# Encodage des variables textuelles 
le = LabelEncoder()
df_target = le.fit_transform(df_target)
df_features.loc[:, 'Departure.TimeStatus.mapped'] = le.fit_transform(df_features['Departure.TimeStatus.mapped'])
df_features.head(10)

# PREPARATION FINALE DES DONNEES 

# Définition des variables features finales 
df_features = df_features[['Departure.TimeStatus.mapped','duree','retard']]

# Séparation des jeux test/entrainement 
X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.30, random_state=42)
# Standardisation des variables numériques 
scaler = StandardScaler()
X_train.loc[:,['duree','retard']] = scaler.fit_transform(X_train[['duree','retard']])
X_test.loc[:,['duree','retard']] = scaler.transform(X_test[['duree','retard']])

# UTILISATION DU MODELE LINEARSVC

model = LinearSVC(dual="auto")

# Entrainement du modèle sur le jeu de données d'entraînement

model.fit(X_train,y_train) 

# Prédiction de la variable cible pour le jeu de données test, ces prédictions sont stockées dans y_pred

y_pred = model.predict(X_test) 

# Evaluation du modèle : à partir de X_test, on prédit la variable cible que l'on compare à y_test

print(model.score(X_test,y_test))

# ANALYSE ET NOTATION DU MODELE
# Evaluation du modèle : à partir de X_test, on prédit la variable cible que l'on compare à y_test
print(model.score(X_test,y_test))

# Classification report 

print(pd.crosstab(y_test,y_pred, rownames=['Realité'], colnames=['Prédiction']))
print(classification_report(y_test, y_pred))

# Importance des paramètres
# Récupérer les coefficients attribués à chaque feature
coefficients = model.coef_[0]
print(model.coef_[0])
# Créer un graphique pour visualiser l'importance relative des features
plt.figure(figsize=(10, 6))
plt.bar(range(len(coefficients)), coefficients)
plt.xticks(range(len(coefficients)), df_features.columns, rotation=45)
plt.xlabel('Features')
plt.ylabel('Coefficient')
plt.title('Importance des variables features dans le modèle LinearSVC')
#plt.show()

#Enregistrer les modèles sous format pickle
joblib.dump(model, 'model.pkl')
joblib.dump(le,'labelencoder.pkl')
joblib.dump(scaler,'scaler.pkl')
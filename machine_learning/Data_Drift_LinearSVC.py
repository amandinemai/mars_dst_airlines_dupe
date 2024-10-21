import mlflow
import mlflow.sklearn
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import ks_2samp
import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction pour surveiller le data drift
def check_data_drift():

    # Charger les nouvelles données depuis le fichier JSON
    with open('flightStatus_Dataset2.json', 'r') as f:
        data = [json.loads(line) for line in f]

    # Création du DataFrame
    df = pd.DataFrame(data)

    # Aplatir les champs imbriqués
    df = pd.json_normalize(data)

    # Selection des colonnes
    df = df[['Departure.TimeStatus.Code', 'Arrival.TimeStatus.Code', 
            'Departure.ScheduledTimeUTC.DateTime', 'Departure.ActualTimeUTC.DateTime', 
            'Arrival.ScheduledTimeUTC.DateTime', 'Arrival.ActualTimeUTC.DateTime']]

    df = df.drop_duplicates()
    df = df.dropna()

    # CALCUL DE LA DUREE DU VOL
    df['duree'] = (pd.to_datetime(df['Arrival.ScheduledTimeUTC.DateTime']) - 
                pd.to_datetime(df['Departure.ScheduledTimeUTC.DateTime']))
    df['duree'] = df['duree'].dt.total_seconds() / 60

    # CALCUL DU RETARD DU VOL
    df['retard'] = (pd.to_datetime(df['Departure.ActualTimeUTC.DateTime']) - 
                    pd.to_datetime(df['Departure.ScheduledTimeUTC.DateTime']))
    df['retard'] = df['retard'].dt.total_seconds() / 60

    df_features = df[['Departure.TimeStatus.Code', 'duree', 'retard']]

    # MAPPING / ENCODAGE

    # Créer un dictionnaire de remplacement
    mapping = {'FE': 0, 'OT': 0, 'DL': 1}
    df['Arrival.TimeStatus.mapped'] = df['Arrival.TimeStatus.Code'].map(mapping).astype(bool)
    df_target = df['Arrival.TimeStatus.mapped']

    # Appliquer le mapping à la colonne cible
    df_features['Departure.TimeStatus.mapped'] = df_features['Departure.TimeStatus.Code'].map(mapping).astype(bool)
    df_features = df_features.drop(columns=['Departure.TimeStatus.Code'])

    # Encode target variable
    le = LabelEncoder()
    df_target = le.fit_transform(df_target)
    df_features['Departure.TimeStatus.mapped'] = le.fit_transform(df_features['Departure.TimeStatus.mapped'])


    df_features = df_features[['Departure.TimeStatus.mapped', 'duree', 'retard']]

    # PREDICTION


    # Charger les artefacts depuis MLflow
    logged_model = 'runs:/f039c5a7754947938a9257707b96a1d3/model' 
    model = mlflow.sklearn.load_model(logged_model)
    X_train = pd.read_csv('X_train.csv')  # Chargez les données d'entraînement sauvegardées
    
    # Normalisation des données d'entraînement
    scaler = StandardScaler()
    X_train[['duree', 'retard']] = scaler.fit_transform(X_train[['duree', 'retard']])
    # Normalisation des nouvelles données
    df_features.loc[:,['duree', 'retard']] = scaler.transform(df_features[['duree', 'retard']])

    # Calculer les métriques de drift
    drift_metrics = {}
    for feature in X_train.columns:
        statistic, p_value = ks_2samp(X_train[feature], df_features[feature])
        drift_metrics[feature] = {'ks_stat': statistic, 'p_value': p_value}

    # Enregistrer les métriques dans MLflow
    mlflow.set_experiment("Data_Drift_flight_status_ML")
    with mlflow.start_run(run_name=f"data_drift_evaluation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"):
        for feature, metrics in drift_metrics.items():
            mlflow.log_metric(f"ks_stat_{feature}", metrics['ks_stat'])
            mlflow.log_metric(f"p_value_{feature}", metrics['p_value'])

            plot_histogram_comparison(X_train, df_features, feature)
            plot_boxplot_comparison(X_train, df_features, feature)
            plot_kde_comparison(X_train, df_features, feature)


        # Enregistrer les métriques de drift
        statistic, p_value = ks_2samp(X_train[feature], df_features[feature])
        mlflow.log_metric(f"ks_stat_{feature}", statistic)
        mlflow.log_metric(f"p_value_{feature}", p_value)

    print("Les métriques de drift ont été enregistrées dans MLflow.")



def plot_histogram_comparison(X_train, df_features, feature_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(X_train[feature_name], bins=50, alpha=0.5, label='Training Data', color='blue')
    sns.histplot(df_features[feature_name], bins=50, alpha=0.5, label='New Data', color='red')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title(f'Distribution Comparison for {feature_name}')

    # Save plot to file
    plot_filename = f'distribution_comparison_{feature_name}.png'
    plt.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)
    plt.show()


def plot_boxplot_comparison(X_train, df_features, feature_name):
    plt.figure(figsize=(10, 6))
    data_to_plot = [X_train[feature_name], df_features[feature_name]]
    plt.boxplot(data_to_plot, labels=['Training Data', 'New Data'])
    plt.ylabel(feature_name)
    plt.title(f'Box Plot Comparison for {feature_name}')

    # Save plot to file
    plot_filename = f'boxplot_comparison_{feature_name}.png'
    plt.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)
    plt.show()

def plot_kde_comparison(X_train, df_features, feature_name):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(X_train[feature_name], shade=True, label='Training Data', color='blue')
    sns.kdeplot(df_features[feature_name], shade=True, label='New Data', color='red')
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.title(f'KDE Plot Comparison for {feature_name}')

    # Save plot to file
    plot_filename = f'kde_comparison_{feature_name}.png'
    plt.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)
    plt.show()

# Define tracking_uri
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Vous pouvez maintenant exécuter cette fonction à intervalles réguliers pour surveiller le data drift
check_data_drift()
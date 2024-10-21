import requests
from config import client_id
from config import client_secret
from config import grant_type

def getapiTocken():

# Envoi de la requête POST
 response = requests.post(
    "https://api.lufthansa.com/v1/oauth/token",
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    data={"client_id": client_id, "client_secret": client_secret, "grant_type": grant_type},)

# Vérification du statut de la réponse
 if response.status_code == 200:
    # Traitement du jeton d'accès si la réponse est réussie
    access_token = response.json()["access_token"]
    return access_token
 else:
    print(f"Erreur : {response.status_code} - {response.text}")
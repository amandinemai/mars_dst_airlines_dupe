import datetime
import requests
from pymongo import MongoClient
import os
import json
import logging
import csv
import pandas as pd
from get_token import getapiTocken

# Configuration d'un logger pour envoyer les logs dans un fichier
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('logfile.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def get_flights():
        access_token = getapiTocken()
        print(access_token)
        code_iata = preparation_code_iata()
        today = datetime.datetime.now().date()
        for i in range(-6, 1):  # Loop for 7 days before today
            date_souhaitee = today + datetime.timedelta(days=i)
            date_depart = date_souhaitee.strftime("%Y-%m-%d") + 'T10:00'
            for row in code_iata:
                call_api(row, date_depart,access_token)


def call_api(code_depart, future_date,access_token):
    try:
        url = build_url(code_depart, future_date)
        
        headers = {
            "Authorization": "Bearer " + access_token,
            "Accept": "application/json"
        }
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            json_data = response.json()
            data = json_data.get("FlightStatusResource", {}).get("Flights", {}).get("Flight", [])

            client = MongoClient("mongodb://admin:password@localhost:27017/")
            db = client["lufthansa"]
            collection = db["StatusAtArrivals"]

            for flight in data:
                collection.insert_one(flight)

            logger.info("Flight data saved to MongoDB.")
            logger.info(url)
            print(url)
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

def build_url(code_arrival, date):
    with open('status_url.json') as config_file:
        config = json.load(config_file)
        base_url = config.get("base_url")
        return f"{base_url}/{code_arrival}/{date}"

def preparation_code_iata():
    df = pd.read_csv('../airport_codes.csv')
    western_europe_countries = ['France', 'Germany', 'United Kingdom', 'Spain', 'Italy', 'Portugal', 'Belgium', 'Netherlands', 'Switzerland', 'Austria', 'Luxembourg', 'Ireland']
    western_europe_airports = df[df['Country'].isin(western_europe_countries)]
    iata_codes = western_europe_airports['IATA Code'].tolist()
    return iata_codes

if __name__ == "__main__":
    get_flights()

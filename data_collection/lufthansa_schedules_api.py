import requests
from pymongo import MongoClient
import os
import json
from get_token import getapiTocken

def call_api():
    access_token = getapiTocken()

    # Build the API endpoint URL
    url = build_url()
    print(build_url())

    # Headers
    headers = {
        "Authorization": "Bearer " + access_token,
        "Accept": "application/json"
    }

    # Make the API request
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Get the JSON data
        json_data = response.json()
        data = json_data["ScheduleResource"]["Schedule"]

        # Connect to MongoDB and insert the flight data
        client = MongoClient("mongodb://admin:password@localhost:27017/")
        db = client["lufthansa"]
        collection = db["Schedule"]
        
        for flight in data:
            collection.insert_one(flight)

        print("Flight data saved to MongoDB.")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        if response.status_code == 401:
            # Call the script to get a valid token
            os.system("python3 get_token.py")
            # Call the API function recursively in case of error
            call_api()

def build_url():
    # Read the configuration from a JSON file
    with open('schedules_url.json') as config_file:
        config = json.load(config_file)
        base_url = config["base_url"]
        departure_code = config["departure_code"]
        arrival_code = config["arrival_code"]
        departure_date = config["departure_date"]

    # Construct the API endpoint URL
    return f"{base_url}/{departure_code}/{arrival_code}/{departure_date}"

# Call the main API function
call_api()
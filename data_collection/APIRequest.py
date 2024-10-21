import requests
import json
import urllib3

def getAPIToken(client_id, client_secret):
	token_url="https://api.lufthansa.com/v1/oauth/token"
	payload={
		"client_id": client_id,
        	"client_secret": client_secret,
        	"grant_type": "client_credentials"
	}
	response = requests.post(
        	token_url,
        	data=payload
    	)
	if response.status_code == 200:
		token = response.json()["access_token"]
		return token
	else:
		print("Failed to obtain access token:", response.text)
		return None

def getAPIResponse(token, url):
	urllib3.disable_warnings()

	headers={'Authorization':'Bearer ' + token}
	response= requests.get(url, headers=headers ,verify=False)

	response_json = response.json()
	return response_json

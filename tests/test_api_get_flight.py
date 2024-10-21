from fastapi.testclient import TestClient
from api.fastapi_main import app
from unittest.mock import patch


client = TestClient(app)

def test_get_flight_valid_flight_number():
    # Mock les données de vol pour simuler une requête réussie
    mock_flight_data = {
        'MarketingCarrier': {'AirlineID': 'LH', 'FlightNumber': '924'},
        'Departure': {
            'AirportCode': 'FRA',
            'ScheduledTimeLocal': {'DateTime': '2024-05-01T07:00'},
            'ScheduledTimeUTC': {'DateTime': '2024-05-01T05:00Z'},
            'ActualTimeLocal': {'DateTime': '2024-05-01T08:01'},
            'ActualTimeUTC': {'DateTime': '2024-05-01T06:01Z'},
            'TimeStatus': {'Code': 'DL', 'Definition': 'Flight Delayed'},
            'Terminal': {'Name': '1', 'Gate': 'B26'}
        },
        'Arrival': {
            'AirportCode': 'LHR',
            'ScheduledTimeLocal': {'DateTime': '2024-05-01T07:40'},
            'ScheduledTimeUTC': {'DateTime': '2024-05-01T06:40Z'},
            'ActualTimeLocal': {'DateTime': '2024-05-01T08:22'},
            'ActualTimeUTC': {'DateTime': '2024-05-01T07:22Z'},
            'TimeStatus': {'Code': 'DL', 'Definition': 'Flight Delayed'},
            'Terminal': {'Name': '2'}
        },
        'Equipment': {'AircraftCode': '32N', 'AircraftRegistration': 'DAINJ'},
        'FlightStatus': {'Code': 'LD', 'Definition': 'Flight Landed'},
        'ServiceType': 'Passenger'
    }

    with patch('api.fastapi_main.collection.find_one', return_value=mock_flight_data):
        response = client.get("/flightInformation", params={"flightNumber": "924"}, headers={"Authorization": "Basic YWRtaW46cGFzc3dvcmQ="})
        assert response.status_code == 200
        assert response.json() == mock_flight_data

def test_get_flight_invalid_flight_number():
    with patch('api.fastapi_main.collection.find_one', return_value=None):
        response = client.get("/flightInformation", params={"flightNumber": "678"}, headers={"Authorization": "Basic YWRtaW46cGFzc3dvcmQ="})
        assert response.status_code == 404
        assert response.json() == {"detail": "Flight not found"}

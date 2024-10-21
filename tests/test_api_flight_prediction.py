import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.fastapi_main import app
from mlflow.tracking import MlflowClient
from fastapi import HTTPException
import pandas as pd
from api.fastapi_users import users
import mlflow.sklearn

client = TestClient(app)

class TestFlightPrediction(unittest.TestCase):

    @patch('api.fastapi_main.authenticate', return_value=True)
    @patch('api.fastapi_main.collection.find_one')
    @patch('api.fastapi_main.mlflow.sklearn.load_model')
    @patch('api.fastapi_main.MlflowClient')
    def test_get_prediction_valid_flight(self, mock_mlflow_client, mock_load_model, mock_find_one, mock_authenticate):
        mock_find_one.return_value = {
            'MarketingCarrier': {'FlightNumber': '924'},
            'Departure': {
                'TimeStatus': {'Code': 'DL'},
                'ScheduledTimeUTC': {'DateTime': '2024-05-01T05:00Z'},
                'ActualTimeUTC': {'DateTime': '2024-05-01T06:01Z'}
            },
            'Arrival': {
                'TimeStatus': {'Code': 'DL'},
                'ScheduledTimeUTC': {'DateTime': '2024-05-01T06:40Z'},
                'ActualTimeUTC': {'DateTime': '2024-05-01T07:22Z'}
            }
        }

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = '1'
        mock_mlflow_client.return_value.get_experiment_by_name.return_value = mock_experiment
        mock_run = MagicMock()
        mock_run.data.params.get.return_value = 'LinearSVC'
        mock_run.info.start_time = 1
        mock_run.info.run_id = '1234'
        mock_mlflow_client.return_value.search_runs.return_value = [mock_run]
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_load_model.return_value = mock_model

        response = client.get("/flightPrediction", params={"flightNumber": "924"}, headers={"Authorization": "Basic YWRtaW46cGFzc3dvcmQ="})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prediction": [1]})

    @patch('api.fastapi_main.authenticate', return_value=True)
    @patch('api.fastapi_main.collection.find_one', return_value=None)
    def test_get_prediction_flight_not_found(self, mock_find_one, mock_authenticate):
        response = client.get("/flightPrediction", params={"flightNumber": "678"}, headers={"Authorization": "Basic YWRtaW46cGFzc3dvcmQ="})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Flight data not found"})

    @patch('api.fastapi_main.authenticate', return_value=True)
    @patch('api.fastapi_main.get_mlflow_experiment')
    def test_get_prediction_no_experiment(self, mock_get_mlflow_experiment, mock_authenticate):
        mock_get_mlflow_experiment.side_effect = ValueError("Experiment 'lufthansa_flight_status_ML' does not exist")
        response = client.get("/flightPrediction", params={"flightNumber": "924"}, headers={"Authorization": "Basic YWRtaW46cGFzc3dvcmQ="})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Experiment 'lufthansa_flight_status_ML' does not exist"})

    @patch('api.fastapi_main.authenticate', return_value=True)
    @patch('api.fastapi_main.get_mlflow_experiment')
    @patch('api.fastapi_main.MlflowClient')
    def test_get_prediction_no_linear_svc_run(self, mock_mlflow_client, mock_get_mlflow_experiment, mock_authenticate):
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = '1'
        mock_get_mlflow_experiment.return_value = mock_experiment
        mock_mlflow_client.return_value.search_runs.return_value = []

        response = client.get("/flightPrediction", params={"flightNumber": "924"}, headers={"Authorization": "Basic YWRtaW46cGFzc3dvcmQ="})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "No runs found for the LinearSVC model"})

if __name__ == "__main__":
    unittest.main()

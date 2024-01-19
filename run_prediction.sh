#!/bin/bash

# Path to input_data.json
INPUT_JSON_FILE="input_data.json"

# URL of API Flask
API_URL="http://localhost:5000/predict"

# Use cURL to send POST request
curl -X POST -H "Content-Type: application/json" -d @"$INPUT_JSON_FILE" "$API_URL"
# curl -X POST -H "Content-Type: application/json" -d @"$INPUT_JSON_FILE" "$API_URL" > output.json

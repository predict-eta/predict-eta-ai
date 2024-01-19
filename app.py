# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model
encode_data = pd.read_pickle('Checkpoints/encode_data.pkl')
encode_dict = joblib.load('Checkpoints/feature_encoder.pkl')
scaling_data = pd.read_pickle('Checkpoints/scaling_data.pkl')
scaling_dict = joblib.load('Checkpoints/feature_scaling.pkl')
model = joblib.load('Checkpoints/xGB.pkl')


# Data Cleaning
def NullClearner(value):
    if(isinstance(value, pd.Series) and (value.dtype in ['float64','int64'])):
         value.fillna(value.mean(),inplace=True)
         return value
    elif(isinstance(value, pd.Series)):
         value.fillna(value.mode()[0],inplace=True)
         return value
    else:
         return value

@app.route('/predict', methods=['POST'])
def predict():
    # Get input by request
    json_data = request.get_json(force=True)
    # Convert json data to dataframe
    input_data = pd.DataFrame.from_dict(json_data)

    x = input_data.columns.to_list()
    for i in x:
        input_data[i] = NullClearner(input_data[i])
    # Replace newline characters with an empty string
    input_data['DEPARTURE'] = input_data['DEPARTURE'].str.replace('\n', '')
    input_data['DEPARTURE'] = input_data['DEPARTURE'].str.replace(' ', '')
    input_data['DEPARTURE'] = input_data['DEPARTURE'].str.split(',').str.get(0)
    input_data['DEPARTURE'] = input_data['DEPARTURE'].str.replace('[^a-zA-Z\s]', '', regex=True).str.strip() # Remove non-alphabetic characters and trim spaces in the 'Column_Name'
    input_data['DEPARTURE'] = input_data['DEPARTURE'].str.upper()

    input_data['DESTINATION'] = input_data['DESTINATION'].str.replace('\n', '')
    input_data['DESTINATION'] = input_data['DESTINATION'].str.replace(' ', '')
    input_data['DESTINATION'] = input_data['DESTINATION'].str.split(',').str.get(0)
    input_data['DESTINATION'] = input_data['DESTINATION'].str.replace('[^a-zA-Z\s]', '', regex=True).str.strip() # Remove non-alphabetic characters and trim spaces in the 'Column_Name'
    input_data['DESTINATION'] = input_data['DESTINATION'].str.upper()
    
    input_data['DEPARTURE'] = input_data['DEPARTURE'].map(encode_dict)
    input_data['DESTINATION'] = input_data['DESTINATION'].map(encode_dict)
    input_data = scaling_dict.transform(input_data)

    prediction = model.predict(input_data)
    # Chuẩn bị kết quả
    result = {'Estimate Time of Arrival (ETA) (days)': round(prediction[0].astype(float))}
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

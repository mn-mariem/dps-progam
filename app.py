from flask import Flask, request, jsonify
from sklearn.preprocessing import OneHotEncoder
import joblib
import pandas as pd 
import numpy as np
import pickle


# Load the model
model_rf = joblib.load('rf_new_model.pkl')

# Load the preprocessing steps
enc = pickle.load(open('encoder.pickle', 'rb'))

app = Flask(__name__)

@app.route("/accident_predictions", methods=["POST"])
def process_json():
    try:
        # Get the input data from the request
        input_data = request.get_json()

        # Access to all features from the input_data
        category = str(input_data['category'])
        type = str(input_data['type'])
        year = input_data['year']
        month = input_data['month']
        
        # Prepare the categorical features
        categorical_features = np.column_stack((type, category,))

        # Use the loaded encoder to one-hot encode the categorical features
        input_encoded = list(enc.transform(categorical_features).toarray()[0])

        # Concatenate the features horizontally
        to_predict = np.array([year, month] + input_encoded)

        # Make predictions using the loaded model
        predictions = model_rf.predict(to_predict.reshape(1, -1))

       # Return results
        return jsonify({'prediction': round(predictions[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=80)

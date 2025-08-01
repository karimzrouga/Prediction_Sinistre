# app.py

from flask import Flask, jsonify, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load your best model
best_model = joblib.load('best_model.pkl')

# Define a prediction endpoint
@app.route('/test', methods=['GET'])
def test():
    return "TEST DONE"
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json  # Assuming JSON input
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data)
        
        # Make predictions
        proba = best_model.predict_proba(input_df)[:, 1]
        pred = best_model.predict(input_df)
        
        # Prepare response
        response = {
            'prediction': pred.tolist(),
            'probability': proba.tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, jsonify, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load your best model
best_model = joblib.load('best_model.pkl')

@app.route('/test', methods=['GET'])
def test():
    return "TEST DONE"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data)
       
        proba = best_model.predict_proba(input_df)[:, 1]
        pred = best_model.predict(input_df)
       
        response = {
            'prediction': pred.tolist(),
            'probability': proba.tolist()
        }
       
        return jsonify(response)
   
    except Exception as e:
        return jsonify({'error': str(e)})

# Cette partie est nécessaire pour Vercel
def vercel_handler(request):
    from flask import Response
    with app.app_context():
        response = app.full_dispatch_request()
        return Response(
            response.get_data(),
            status=response.status_code,
            headers=dict(response.headers)
        )

from flask import Flask, jsonify, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Chargement du modèle avec debug
print("Tentative de chargement du modèle...")
try:
    # Afficher le répertoire courant pour debug
    print(f"Répertoire courant: {os.getcwd()}")
    print(f"Contenu du répertoire: {os.listdir('.')}")
    
    # Essayer différents chemins
    possible_paths = [
        'best_model.pkl',
        '../best_model.pkl',
        './best_model.pkl'
    ]
    
    best_model = None
    for path in possible_paths:
        try:
            if os.path.exists(path):
                best_model = joblib.load(path)
                print(f"Modèle chargé depuis: {path}")
                break
        except Exception as e:
            print(f"Échec pour {path}: {e}")
            continue
            
    if best_model is None:
        print("Aucun modèle trouvé")
        
except Exception as e:
    print(f"Erreur générale: {e}")
    best_model = None

@app.route('/')
def home():
    return jsonify({
        "message": "API de prédiction ML",
        "status": "running",
        "model_loaded": best_model is not None,
        "endpoints": {
            "/test": "GET - Test de l'API",
            "/predict": "POST - Faire une prédiction"
        }
    })

@app.route('/test')
def test():
    return jsonify({
        "message": "TEST DONE", 
        "status": "success",
        "model_status": "loaded" if best_model else "not_loaded"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if best_model is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
            
        input_data = request.json
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
            
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame(input_data)
       
        # Make predictions
        proba = best_model.predict_proba(input_df)[:, 1]
        pred = best_model.predict(input_df)
       
        response = {
            'prediction': pred.tolist(),
            'probability': proba.tolist(),
            'status': 'success'
        }
       
        return jsonify(response)
   
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Export pour Vercel
def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    app.run(debug=True)

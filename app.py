from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = joblib.load('best_wine_quality_model.pkl')

# Initialize the Flask application
app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return "Wine Quality Prediction Model API"

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Convert data into DataFrame
    data_df = pd.DataFrame([data])

    # Rescale features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_df)

    # Make predictions
    prediction = model.predict(data_scaled)

    # Return the result as JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

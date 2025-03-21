from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS (to allow cross-origin requests)
CORS(app)

# Load the model (make sure the model is in the same directory or specify the correct path)
model = tf.keras.models.load_model('nse_lstm_model_fixed.h5')

@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from request
        data = request.get_json()
        input_data = np.array(data['input'])  # shape must be [1, 60, 7]

        # Validate input shape
        if input_data.shape != (1, 60, 7):
            return jsonify({"error": "Expected input shape [1, 60, 7]"}), 400

        # Make prediction
        prediction = model.predict(input_data)
        predicted_price = float(prediction[0][0])

        # Return predicted price and reason
        return jsonify({
            "predicted_price": predicted_price,
            "reason": "Predicted using 60 days of technical input with LSTM"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)  # Render expects port 80

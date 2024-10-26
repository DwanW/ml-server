from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

# Load pre-trained sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json.get('text')
        if not data:
            return jsonify({'error': 'No text provided'}), 400

        result = sentiment_model(data)[0]
        response = {
            'label': result['label'],
            'confidence': result['score']
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request
import joblib
from backend.image_text_processor import preprocess_text, preprocess_image

app = Flask(__name__)

text_model = joblib.load('backend/text_model.pkl')
image_model = joblib.load('backend/image_model.pkl')
tfidf_vectorizer = joblib.load('backend/tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    image = request.files.get('image')

    result = {}

    if text:
        cleaned_text = preprocess_text(text)
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        text_prediction = text_model.predict(vectorized_text)[0]
        result['Text Sentiment'] = text_prediction

    if image:
        image_array = preprocess_image(image)
        image_prediction = image_model.predict(image_array)[0]
        result['Image Sentiment'] = image_prediction

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

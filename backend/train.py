# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import build_text_model, build_image_model, extract_image_features
import os
import joblib

# Load and clean data
data_path = os.path.join('..', 'dataset', 'labels.csv')
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace in column names

# Drop rows with missing 'text_corrected' or 'overall_sentiment'
data = data.dropna(subset=['text_corrected', 'overall_sentiment'])

# Ensure text is string type
data['text_corrected'] = data['text_corrected'].astype(str)

# Map sentiment to numeric labels
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2, 'very_positive': 3}
data['overall_sentiment'] = data['overall_sentiment'].map(sentiment_map)

# Drop rows with invalid sentiment (e.g., not in map)
data = data.dropna(subset=['overall_sentiment'])

# Extract texts and labels
texts = data['text_corrected'].values
labels = data['overall_sentiment'].astype(int).values

# Train text model
vectorizer, text_clf = build_text_model()
text_features = vectorizer.fit_transform(texts).toarray()
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    text_features, labels, test_size=0.2, random_state=42
)
text_clf.fit(X_train_text, y_train_text)

# Train image model
image_folder = os.path.join('..', 'dataset', 'images')
image_paths = [
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.endswith(('.jpg', '.png'))
]

# Limit to number of labels (assuming images and text have same order/count)
img_features = np.array([extract_image_features(path) for path in image_paths[:len(labels)]])
scaler, img_clf = build_image_model()
scaled_features = scaler.fit_transform(img_features)
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    scaled_features, labels, test_size=0.2, random_state=42
)
img_clf.fit(X_train_img, y_train_img)

# Save models
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(text_clf, 'text_clf.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(img_clf, 'img_clf.joblib')

print("✅ Training completed. Models saved successfully.")

#python train.py
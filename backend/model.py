from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def build_text_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def build_image_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def extract_image_features(image_array):
    return image_array  # Already flattened in preprocessing

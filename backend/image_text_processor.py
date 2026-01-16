import cv2
import numpy as np
import re
import string
from PIL import Image

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_image(image_file):
    image = Image.open(image_file).convert('L')
    image = image.resize((64, 64))
    return np.array(image).flatten().reshape(1, -1)

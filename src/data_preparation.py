# data_preparation.py
import numpy as np
from datasets import load_dataset
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image

def load_and_preprocess_data(target_size=(224, 224)):
    """
    Fungsi untuk memuat dataset TrashNet dari Hugging Face dan melakukan
    preprocessing (resize gambar dan normalisasi).
    """
    # Muat dataset
    dataset = load_dataset("garythung/trashnet")
    data_train = dataset['train']
    
    images = []
    labels = []
    
    # Proses setiap gambar
    for example in data_train:
        img = example['image'].resize(target_size)  # Resize gambar
        img = np.array(img) / 255.0  # Normalisasi gambar
        images.append(img)
        labels.append(example['label'])
    
    # Convert to numpy array
    X = np.array(images)
    y = np.array(labels)
    
    # One-hot encode label
    num_classes = len(set(y))
    y = to_categorical(y, num_classes)
    
    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, num_classes 

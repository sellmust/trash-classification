from tensorflow.keras import layers, models
from wandb.integration.keras import WandbCallback
from tensorflow.keras.callbacks import ModelCheckpoint
import wandb
from huggingface_hub import HfApi, HfFolder

def build_cnn_model(input_shape, num_classes):
    """
    Fungsi untuk membangun model CNN dasar.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(X_train, y_train, X_test, y_test, num_classes, project_name, entity_name):
    """
    Fungsi untuk melatih model CNN dan mengirimkan log ke wandb.
    """
    # Inisialisasi wandb
    wandb.init(project=project_name, entity=entity_name, name="cnn_baseline")
    
    # Bangun model
    cnn_model = build_cnn_model(X_train.shape[1:], num_classes)
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Melatih model
    history = cnn_model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[
            WandbCallback(log_graph=False),  # Nonaktifkan penyimpanan otomatis Wandb
            ModelCheckpoint(filepath="trashnet_model.keras", save_best_only=True)  # Simpan model terbaik
        ]
    )
    
    # Simpan model
    model_path = "trashnet_model"
    cnn_model.save(model_path)
    print(f"Model disimpan di {model_path}")
    
    return cnn_model, history, model_path

def upload_to_huggingface(model_path, repo_id, hf_token):
    """
    Fungsi untuk mengunggah model ke Hugging Face Hub.
    """
    api = HfApi()
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        token=hf_token
    )
    print(f"Model berhasil diunggah ke Hugging Face Hub: {repo_id}")

# Contoh penggunaan fungsi
if __name__ == "__main__":
    import numpy as np
    from tensorflow.keras.utils import to_categorical
    
    # Contoh data dummy (ganti dengan data sebenarnya)
    input_shape = (224, 224, 3)
    num_classes = 6
    X_train = np.random.rand(100, 224, 224, 3)
    y_train = to_categorical(np.random.randint(0, num_classes, 100), num_classes)
    X_test = np.random.rand(20, 224, 224, 3)
    y_test = to_categorical(np.random.randint(0, num_classes, 20), num_classes)
    
    # Inisialisasi parameter
    project_name = "trashnet-classification"
    entity_name = "sellyymus"  # Ganti dengan username Wandb Anda
    repo_id = "username/trash-classification"  # Ganti dengan repository Hugging Face Anda
    hf_token = "9e0e4a9ebaefa4f8d16f56ed9bccd52164faf4c9"  # Ganti dengan token API Hugging Face Anda
    
    # Latih model
    model, history, model_path = train_model(X_train, y_train, X_test, y_test, num_classes, project_name, entity_name)
    
    # Unggah model ke Hugging Face Hub
    upload_to_huggingface(model_path, repo_id, hf_token)
 

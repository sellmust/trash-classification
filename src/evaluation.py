# evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_training(history):
    """
    Fungsi untuk memplot grafik akurasi dan loss selama pelatihan.
    """
    plt.figure(figsize=(12, 4))
    # Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy")
    plt.legend()
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Loss")
    plt.legend()
    plt.show()

def evaluate_model(cnn_model, X_test, y_test):
    """
    Fungsi untuk mengevaluasi model dan menampilkan hasil.
    """
    # Evaluasi model
    loss, acc = cnn_model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")

    # Confusion Matrix
    y_pred = np.argmax(cnn_model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(y_test.shape[1]))
    disp.plot(cmap='viridis')
    plt.title("Confusion Matrix")
    plt.show()

 

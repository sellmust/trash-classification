# utils.py
import numpy as np
import matplotlib.pyplot as plt

def show_sample_images(X, y, num_classes=6):
    """
    Fungsi untuk menampilkan contoh gambar dari dataset.
    """
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
    for i in range(num_classes):
        idx = np.where(y.argmax(axis=1) == i)[0][0]
        axes[i].imshow(X[idx])
        axes[i].axis('off')
        axes[i].set_title(f"Class {i}")
    plt.show()

def plot_class_distribution(y_train):
    """
    Fungsi untuk memplot distribusi kelas dalam data latih.
    """
    sns.countplot(x=y_train.argmax(axis=1), palette="Set2")
    plt.title("Distribusi Kelas dalam Data Latih")
    plt.show() 

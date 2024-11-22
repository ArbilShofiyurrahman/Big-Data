import streamlit as st
from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

# Konfigurasi Flask
app = Flask(__name__)

# Direktori kelas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_DIRS = {
    "Arborio": os.path.join(BASE_DIR, "Arborio"),
    "Basmati": os.path.join(BASE_DIR, "Basmati"),
    "Ipsala": os.path.join(BASE_DIR, "Ipsala"),
    "Jasmine": os.path.join(BASE_DIR, "Jasmine"),
    "Karacadag": os.path.join(BASE_DIR, "Karacadag"),
}

# Memuat model pre-trained MobileNetV2 dan menambahkan layer klasifikasi
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Tidak melatih model dasar

# Menambahkan lapisan klasifikasi
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 kelas
])

# Mengkompilasi model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fungsi untuk mengklasifikasikan gambar
def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Menyesuaikan ukuran gambar
    img_array = image.img_to_array(img)  # Mengubah gambar menjadi array
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch
    img_array = img_array / 255.0  # Normalisasi gambar
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)  # Menemukan kelas dengan skor tertinggi
    class_labels = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]  # Daftar kelas
    return class_labels[class_idx]  # Mengembalikan nama kelas yang diprediksi

# Halaman Utama
@app.route('/')
def home():
    examples = {}
    for class_name, path in CLASS_DIRS.items():
        # Ambil satu contoh gambar dari masing-masing kelas
        images = [img for img in os.listdir(path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            examples[class_name] = os.path.join(class_name, images[0])
    return render_template('index.html', examples=examples)

# Halaman Upload
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(BASE_DIR, filename)
            file.save(file_path)
            
            # Prediksi kelas gambar menggunakan CNN
            predicted_class = classify_image(file_path)

            # Hapus file upload setelah prediksi (opsional)
            os.remove(file_path)
            
            return redirect(url_for('result', predicted_class=predicted_class))
    return render_template('upload.html')

# Halaman Hasil
@app.route('/result/<predicted_class>')
def result(predicted_class):
    return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)

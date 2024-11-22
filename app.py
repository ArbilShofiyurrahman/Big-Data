import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Menyusun model CNN pre-trained
@st.cache_resource
def load_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Tidak melatih model dasar
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(5, activation='softmax')  # 5 kelas
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# Daftar kelas yang akan digunakan untuk klasifikasi
CLASS_LABELS = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

# Fungsi untuk mengklasifikasikan gambar
def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Menyesuaikan ukuran gambar
    img_array = image.img_to_array(img)  # Mengubah gambar menjadi array
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch
    img_array = img_array / 255.0  # Normalisasi gambar
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)  # Menemukan kelas dengan skor tertinggi
    return CLASS_LABELS[class_idx]  # Mengembalikan nama kelas yang diprediksi

# Aplikasi Streamlit
def main():
    st.title("Klasifikasi Citra Padi")
    st.write("Aplikasi ini dapat mengklasifikasikan gambar padi ke dalam 5 kelas: Arborio, Basmati, Ipsala, Jasmine, Karacadag.")

    # Membuat pilihan untuk mengunggah gambar
    uploaded_file = st.file_uploader("Pilih gambar padi untuk diklasifikasikan", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Menyimpan file sementara untuk prediksi
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Klasifikasi gambar
        if st.button("Klasifikasikan Gambar"):
            predicted_class = classify_image("uploaded_image.jpg")
            st.write(f"Prediksi kelas gambar: {predicted_class}")

            # Hapus file sementara
            os.remove("uploaded_image.jpg")

if __name__ == "__main__":
    main()

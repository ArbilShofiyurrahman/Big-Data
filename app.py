import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Menyusun model CNN pre-trained dan fine-tuning
@st.cache_resource
def load_model():
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Menonaktifkan pelatihan model dasar
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
    return predictions[0]  # Mengembalikan probabilitas dari setiap kelas

# Aplikasi Streamlit
def main():
    st.title("Klasifikasi Citra Padi")
    st.write("Aplikasi ini dapat mengklasifikasikan gambar padi ke dalam 5 kelas: Arborio, Basmati, Ipsala, Jasmine, Karacadag.")
    
    # Menampilkan contoh gambar tiap kelas dari folder masing-masing
    st.write("Contoh gambar untuk tiap kelas:")
    
    # Menampilkan gambar per kelas dari folder yang sesuai
    image_paths = {}
    for class_name in CLASS_LABELS:
        folder_path = class_name
        # Ambil gambar pertama dari folder untuk setiap kelas
        class_images = os.listdir(folder_path)
        first_image_path = os.path.join(folder_path, class_images[0])  # Mengambil gambar pertama
        image_paths[class_name] = first_image_path

    cols = st.columns(5)  # Membuat 5 kolom untuk menampilkan gambar dalam 1 baris
    for i, class_name in enumerate(CLASS_LABELS):
        with cols[i]:
            st.image(image_paths[class_name], caption=class_name, use_column_width=True)

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
            predictions = classify_image("uploaded_image.jpg")
            predicted_class = CLASS_LABELS[np.argmax(predictions)]
            st.write(f"Prediksi kelas gambar: {predicted_class}")

            # Menyimpan hasil probabilitas untuk visualisasi
            st.session_state["predictions"] = predictions

            # Hapus file sementara
            os.remove("uploaded_image.jpg")

    # Menu untuk Insight Visualisasi
    if "predictions" in st.session_state:
        
        # Tombol untuk menampilkan insight
        if st.button("Tampilkan Insight"):
            predictions = st.session_state["predictions"]
            
            # Membuat grafik dengan ukuran lebih kecil
            fig, ax = plt.subplots(figsize=(2, 1.5))  # Ubah ukuran dengan figsize (lebar, tinggi)
            ax.bar(CLASS_LABELS, predictions, color="skyblue")
            ax.set_title("Probabilitas Tiap Kelas", fontsize=5)
            ax.set_ylabel("Probabilitas", fontsize=4)
            ax.set_xlabel("Kelas", fontsize=4)
            ax.tick_params(axis='x', labelsize=4)  # Ukuran label sumbu x lebih kecil
            ax.tick_params(axis='y', labelsize=4)  # Ukuran label sumbu y lebih kecil
            
            # Tampilkan grafik dalam Streamlit
            st.pyplot(fig)

if __name__ == "__main__":
    main()

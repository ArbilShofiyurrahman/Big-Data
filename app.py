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
    class_idx = np.argmax(predictions)  # Menemukan kelas dengan skor tertinggi
    return CLASS_LABELS[class_idx], predictions[0]  # Mengembalikan nama kelas dan skor probabilitas

# Fungsi untuk menampilkan insight visualisasi hasil klasifikasi
def display_insight(results):
    st.subheader("Visualisasi Hasil Klasifikasi")
    st.write("Grafik di bawah ini menunjukkan distribusi hasil klasifikasi pada data yang telah dianalisis:")
    
    # Membuat grafik
    labels = list(results.keys())
    values = list(results.values())
    
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax.set_title("Distribusi Hasil Klasifikasi")
    ax.set_xlabel("Kelas Padi")
    ax.set_ylabel("Jumlah Prediksi")
    
    st.pyplot(fig)

# Aplikasi Streamlit
def main():
    st.title("Klasifikasi Citra Padi")
    st.write("Aplikasi ini dapat mengklasifikasikan gambar padi ke dalam 5 kelas: Arborio, Basmati, Ipsala, Jasmine, Karacadag.")
    
    # Menampilkan contoh gambar tiap kelas dari folder masing-masing
    st.write("Contoh gambar untuk tiap kelas:")
    image_paths = {}
    for class_name in CLASS_LABELS:
        folder_path = class_name
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            class_images = os.listdir(folder_path)
            if class_images:
                first_image_path = os.path.join(folder_path, class_images[0])  # Mengambil gambar pertama
                image_paths[class_name] = first_image_path

    cols = st.columns(5)  # Membuat 5 kolom untuk menampilkan gambar dalam 1 baris
    for i, class_name in enumerate(CLASS_LABELS):
        if class_name in image_paths:
            with cols[i]:
                st.image(image_paths[class_name], caption=class_name, use_column_width=True)

    # Membuat pilihan untuk mengunggah gambar
    uploaded_file = st.file_uploader("Pilih gambar padi untuk diklasifikasikan", type=["jpg", "png", "jpeg"])
    
    # Menyimpan hasil klasifikasi untuk visualisasi
    classification_results = {label: 0 for label in CLASS_LABELS}
    
    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Menyimpan file sementara untuk prediksi
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Klasifikasi gambar
        if st.button("Klasifikasikan Gambar"):
            predicted_class, prediction_scores = classify_image("uploaded_image.jpg")
            st.write(f"Prediksi kelas gambar: {predicted_class}")
            st.write(f"Probabilitas tiap kelas:")
            for i, label in enumerate(CLASS_LABELS):
                st.write(f"{label}: {prediction_scores[i]:.2f}")
            
            # Tambahkan hasil prediksi ke dalam hasil klasifikasi
            classification_results[predicted_class] += 1

            # Hapus file sementara
            os.remove("uploaded_image.jpg")

    # Menampilkan grafik distribusi hasil klasifikasi
    display_insight(classification_results)

if __name__ == "__main__":
    main()

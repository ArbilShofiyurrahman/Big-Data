import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

# Daftar kata kunci untuk setiap kategori
keywords = {
    "wisata religi": [
        "masjid", "pura", "vihara", "upacara", "ziarah", "makam", "habib", "ramadhan", "religi",
        "sunan", "candi", "agama", "islam", "borobudur", "al-qur'an", "waisak", "jemaah", "ibadah",
        "kubah", "gereja", "patung", "rosario", "mezbah", "biara", "meditasi", "thirtha", "haji",
        "umrah", "doa", "yasinan", "istighosah", "tahlilan", "tadarus", "misa", "kelenteng"
    ],
    "wisata alam": [
        "gunung", "pantai", "hutan", "sungai", "air terjun", "camping", "dieng", "baduy", "labuan bajo",
        "ancol", "taman nasional", "gunung bromo", "raja ampat", "pulau komodo", "tanjung puting", "kawah ijen",
        "semeru", "merapi", "rinjani", "taman laut", "taman safari", "suaka margasatwa", "goa", "cagar alam",
        "danau", "terumbu karang", "pesisir", "sabana", "bukit", "lembah", "sawah", "perkemahan",
        "tebing", "mangrove", "sumber air panas"
    ],
    "wisata buatan": [
        "taman bermain", "museum", "kebun binatang", "monumen", "pasar", "hotel", "restoran",
        "waterpark", "theme park", "galeri seni", "taman kota", "gedung konser", "teater", "sirkus",
        "kebun raya", "taman bunga", "kolam renang", "kebun teh", "kebun kopi", "jembatan", "stadion",
        "arena olahraga", "perpustakaan", "bioskop", "pusat sains", "observatorium", "taman hiburan",
        "taman edukasi", "pameran", "galeri", "pusat kerajinan", "toko suvenir", "spa", "resort", "lapangan golf"
    ]
}

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Load dataset
dataset_path = "https://gist.githubusercontent.com/ArbilShofiyurrahman/599d79fc45a06ae40a750f558879392e/raw/2dad7e7cdaca0964bfc85c3efe17b1c0c94a3f70/setelahpreprocess.csv"  # Update with your dataset path
dataset = pd.read_csv(dataset_path)

# Preprocess dataset
dataset['Konten'] = dataset['Konten'].apply(preprocess_text)

# Fungsi untuk ekstraksi fitur kata kunci
def extract_keyword_features(text):
    features = []
    for category, keywords_list in keywords.items():
        features.append(any(keyword in text for keyword in keywords_list))
    return features

# Ekstraksi fitur kata kunci untuk setiap konten
keyword_features = dataset['Konten'].apply(extract_keyword_features)
keyword_features = np.array(keyword_features.tolist())

# Split dataset into X and y
x_data = dataset['Konten']
y_data = dataset['LabelId']

# Vectorize text data
tfidf = TfidfVectorizer(max_features=5000)
x_data_vectorized = tfidf.fit_transform(x_data)

# Menggabungkan fitur TF-IDF dengan fitur kata kunci
x_data_combined = sp.hstack([x_data_vectorized, keyword_features])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_data_combined, y_data, test_size=0.2, random_state=0, shuffle=True)

# Initialize and train KNN model
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(x_train, y_train)

# Streamlit app
def main():
    st.title("Klasifikasi Berita Pariwisata")
    st.subheader("Selamat datang di aplikasi klasifikasi berita pariwisata!")
    st.markdown("""
    ### Informasi Aplikasi
    Aplikasi ini dirancang untuk mengklasifikasikan berita pariwisata ke dalam tiga kategori utama:
    1. **Wisata Religi**: Berita terkait tempat ibadah, kegiatan keagamaan, dan tempat ziarah.
    2. **Wisata Alam**: Berita tentang keindahan alam seperti gunung, pantai, hutan, dan danau.
    3. **Wisata Buatan**: Berita yang berhubungan dengan tempat-tempat buatan manusia seperti taman bermain, museum, dan pasar.

    Masukkan teks berita pariwisata di bawah ini untuk melihat kategori yang sesuai.
    """)
    st.subheader("Masukkan Berita Pariwisata di bawah:")

    # Text input for news article
    article_text = st.text_area("Masukan Konten", "")

    if st.button("Kategori Kan Sekarang"):
        # Preprocess text
        processed_text = preprocess_text(article_text)
        # Vectorize text
        text_vectorized = tfidf.transform([processed_text])
        # Extract keyword features
        keyword_features_input = np.array(extract_keyword_features(processed_text)).reshape(1, -1)
        # Combine features
        text_combined = sp.hstack([text_vectorized, keyword_features_input])
        # Predict category
        prediction = knn_classifier.predict(text_combined)
        # Map prediction to category
        categories = {0: "Wisata Religi", 1: "Wisata Alam", 2: "Wisata Buatan"}
        predicted_category = categories.get(prediction[0], "Unknown")
        # Display result
        st.write("Berita Ini Termasuk Kategori:", predicted_category)

        

if __name__ == "__main__":
    main()

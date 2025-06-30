import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# NLTK veri setlerini indir (ilk çalıştırmada)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Model ve TF-IDF yükle
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Temizleme fonksiyonu
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# Streamlit arayüzü
st.set_page_config(page_title="Kripto Haber Analiz", page_icon="🧠")
st.title("🧠 Kripto Haber Güvenilirlik Tahmini")

user_input = st.text_input("🔍 Bir kripto haber başlığı girin:")

if st.button("📊 Tahmin Et"):
    if user_input.strip() == "":
        st.warning("Lütfen bir haber başlığı girin.")
    else:
        clean = preprocess(user_input)
        vectorized = vectorizer.transform([clean])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("✅ Bu haber **OLUMLU / GÜVENİLİR** olabilir.")
        else:
            st.error("❌ Bu haber **OLUMSUZ / YANILTICI** olabilir.")

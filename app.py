
import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def detect_coin(text):
    coins = ["BTC", "ETH", "XRP", "SOL", "ADA", "BNB", "DOGE", "LTC", "MATIC", "DOT"]
    for coin in coins:
        if coin.lower() in text.lower() or coin in text:
            return coin
    return "Belirsiz"


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


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
        probability = model.predict_proba(vectorized)[0]
        confidence_score = round(probability[1] * 100, 2)  # sınıf 1 (up) için

        label = "up" if prediction == 1 else "down"
        coin = detect_coin(user_input)

        st.markdown(f"### 📌 Başlık: {user_input}")
        st.markdown(f"### 📊 Güven skoru: %{confidence_score}")
        st.markdown(f"### 📈 Tahmin: **{label.upper()}**")
        st.markdown(f"### 🪙 Coin: {coin}")

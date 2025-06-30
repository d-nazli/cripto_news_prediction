"""
import pickle
import re
import string
import nltk

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from save_result import save_result, detect_coin

# Gerekli NLTK verilerini indir (bir kereye mahsus)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 📦 Kayıtlı model ve vektörleştiriciyi yükle
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 🔧 Temizleme fonksiyonu (model eğitimiyle aynı olmalı!)
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# 🧪 Kullanıcıdan haber başlığı al
user_input = input("\n📝 Lütfen haber başlığını girin: ")

# Temizle ve TF-IDF vektörüne çevir
clean_text = preprocess(user_input)
vectorized = vectorizer.transform([clean_text])

prediction = model.predict(vectorized)[0]
probability = model.predict_proba(vectorized)[0]
confidence = round(probability[1] * 100, 2)  # Etiket 1 (olumlu/güvenilir) için yüzde

# ✅ Sonuç yazdır
print("\n📌 Girilen Başlık:", user_input)
print(f"📊 Model güven skoru: %{confidence}")
if prediction == 1:
    print("✅ Bu haber OLUMLU / GÜVENİLİR olabilir.")
else:
    print("❌ Bu haber OLUMSUZ / YANILTICI olabilir.")
"""
"""
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from save_result import save_result, detect_coin  # JSON kaydı için

# Gerekli NLTK verilerini indir (bir kereye mahsus)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 📦 Kayıtlı model ve vektörleştiriciyi yükle
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 🔧 Temizleme fonksiyonu (model eğitimiyle aynı olmalı!)
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# 🧪 Kullanıcıdan haber başlığı al
user_input = input("\n📝 Lütfen haber başlığını girin: ")

# 🔍 Temizle ve vektöre dönüştür
clean_text = preprocess(user_input)
vectorized = vectorizer.transform([clean_text])

# 🤖 Tahmin ve doğruluk oranı
prediction = model.predict(vectorized)[0]
probability = model.predict_proba(vectorized)[0]
confidence = round(probability[1] * 100, 2)  # Etiket 1 (olumlu/güvenilir) için yüzde

# 🧠 Tahmine göre sonuç etiketi
prediction_label = "YÜKSELECEK" if prediction == 1 else "DÜŞECEK"
affected_coin = detect_coin(user_input)

# ✅ Ekrana yazdır
print("\n📌 Girilen Başlık:", user_input)
print(f"📊 Model güven skoru: %{confidence}")
if prediction == 1:
    print("✅ Bu haber OLUMLU / GÜVENİLİR olabilir.")
else:
    print("❌ Bu haber OLUMSUZ / YANILTICI olabilir.")

# 📁 JSON dosyasına sonucu kaydet
save_result(user_input, confidence, affected_coin, prediction_label)
"""


import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from save_result import save_result, detect_coin  # JSON kaydı için

# Gerekli NLTK verilerini indir (ilk çalıştırmada gerekebilir)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 📦 Eğitilmiş model ve TF-IDF vektörleştiriciyi yükle
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 🔧 Haber başlığını temizleyen fonksiyon
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# 🧪 Kullanıcıdan haber başlığı al
user_input = input("\n📝 Lütfen haber başlığını girin: ")

# 🧼 Temizle → Vektörleştir → Tahmin et
clean_text = preprocess(user_input)
vectorized = vectorizer.transform([clean_text])
prediction = model.predict(vectorized)[0]
probability = model.predict_proba(vectorized)[0]
confidence_score = round(probability[1] * 100, 2)  # sınıf 1 (güvenilir) için oran

# 🔄 up / down etiketini belirle
prediction_label = "up" if prediction == 1 else "down"

# 🪙 Coin sembolünü belirle
affected_coin = detect_coin(user_input)

# 📢 Sonucu yazdır
print("\n📌 Başlık:", user_input)
print(f"📊 Güven skoru: %{confidence_score}")
print(f"📈 Tahmin: {prediction_label}")
print(f"🪙 Coin: {affected_coin}")

# 💾 JSON dosyasına sonucu kaydet
save_result(user_input, confidence_score, affected_coin, prediction_label)

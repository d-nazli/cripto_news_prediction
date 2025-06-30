"""
import pandas as pd           # CSV dosyasını okumak ve işlemek için
import re                     # Metin temizliği (regex ile)
import string                 # Noktalama işaretlerini tanımak için
import nltk                   # Doğal dil işleme araçları

# NLTK araçlarını içe aktar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Gerekli NLTK veri setlerini indir
nltk.download('punkt')        # kelime ayırıcı
nltk.download('stopwords')    # gereksiz kelimeler listesi
nltk.download('wordnet')      # kök bulucu sözlük
nltk.download('omw-1.4')      # kök bulma için ekstra veri

# CSV dosyasını oku
df = pd.read_csv("cryptonews.csv")

# Biz haber başlığı üzerinde çalışacağız
df['raw_text'] = df['title']

# Temizleme fonksiyonu
def preprocess(text):
    text = text.lower()  # Küçük harf yap
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    text = text.translate(str.maketrans('', '', string.punctuation))  # Noktalama kaldır
    tokens = word_tokenize(text)  # Kelimelere ayır
    tokens = [w for w in tokens if w not in stopwords.words('english')]  # Stopword temizle
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]  # Kök bul
    return " ".join(tokens)

# Temizleme işlemini tüm haberlere uygula
df['clean_text'] = df['raw_text'].apply(preprocess)

# Sonuçları göster
print(df[['raw_text', 'clean_text']].head())

# ✅ Temizlenmiş veriyi kaydet (etiketleme aşamasında kullanılacak)
df.to_csv("cleaned_news.csv", index=False)
print("✅ Temizlenmiş veri 'cleaned_news.csv' dosyasına kaydedildi.")
"""
import pandas as pd
import re
import string
import nltk

# NLTK araçlarını içe aktar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Gerekli NLTK veri setlerini indir (ilk çalıştırmada)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 📌 1. CSV dosyasını oku
df = pd.read_csv("cryptonews.csv")

# 📌 2. Başlık sütununu kontrol et, boşları sil
df = df.dropna(subset=['title'])
df['raw_text'] = df['title']

# 📌 3. Temizleme fonksiyonu
def preprocess(text):
    text = str(text).lower()  # Küçük harfe çevir
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    text = text.translate(str.maketrans('', '', string.punctuation))  # Noktalama kaldır
    tokens = word_tokenize(text)  # Kelimelere ayır
    tokens = [w for w in tokens if w not in stopwords.words('english')]  # Stopword temizle
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]  # Kök bul
    return " ".join(tokens)

# 📌 4. Temizleme işlemini tüm metinlere uygula
df['clean_text'] = df['raw_text'].apply(preprocess)

# 📌 5. Temizleme sonrası oluşan boş metinleri çıkar
df = df.dropna(subset=['clean_text'])

# 📌 6. Sonuçları göster
print("\nTemizlenmiş örnek veri:")
print(df[['raw_text', 'clean_text']].head())

# 📌 7. Yeni dosya olarak kaydet
df.to_csv("cleaned_news.csv", index=False)
print("\n✅ Temizlenmiş veri 'cleaned_news.csv' dosyasına kaydedildi.")

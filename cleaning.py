
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


df = pd.read_csv("cryptonews.csv")


df = df.dropna(subset=['title'])
df['raw_text'] = df['title']


def preprocess(text):
    text = str(text).lower()  # Küçük harfe çevir
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    text = text.translate(str.maketrans('', '', string.punctuation))  # Noktalama kaldır
    tokens = word_tokenize(text)  # Kelimelere ayır
    tokens = [w for w in tokens if w not in stopwords.words('english')]  # Stopword temizle
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]  # Kök bul
    return " ".join(tokens)


df['clean_text'] = df['raw_text'].apply(preprocess)


df = df.dropna(subset=['clean_text'])


print("\nTemizlenmiş örnek veri:")
print(df[['raw_text', 'clean_text']].head())


df.to_csv("cleaned_news.csv", index=False)
print("\n✅ Temizlenmiş veri 'cleaned_news.csv' dosyasına kaydedildi.")

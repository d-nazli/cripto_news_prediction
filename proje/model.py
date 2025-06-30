import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 1. Veriyi yükle
df = pd.read_csv("labeled_news.csv")

# 2. Girdi (X) ve çıktı (y) belirle
X = df['clean_text']
y = df['label']

# 3. TF-IDF vektörleştirici
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# 4. Eğitim/Test ayır
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 5. Naive Bayes modeli ile eğit
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Test setiyle başarı değerlendirmesi
y_pred = model.predict(X_test)

print("\n📊 Model Başarı Raporu:\n")
print(classification_report(y_test, y_pred))
print(f"✅ Doğruluk Oranı: {accuracy_score(y_test, y_pred):.2%}")

# 7. Model ve vektörleştiriciyi kaydet
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n💾 Model ve vektörleştirici dosyaları kaydedildi: trained_model.pkl, vectorizer.pkl")

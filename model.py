import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("labeled_news.csv")

X = df['clean_text']
y = df['label']

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n📊 Model Başarı Raporu:\n")
print(classification_report(y_test, y_pred))
print(f"✅ Doğruluk Oranı: {accuracy_score(y_test, y_pred):.2%}")


joblib.dump(model, "trained_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

cm = confusion_matrix(y_test, y_pred)


class_names = sorted(df['label'].unique())


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.title("Confusion Matrix - Logistic Regresyon")
plt.tight_layout()
plt.show()

print("\n💾 Model ve vektörleştirici dosyaları kaydedildi: trained_model.pkl, vectorizer.pkl")

import json
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


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


with open("unified_news_analysis.json", "r", encoding="utf-8") as f:
    data = json.load(f)


for i, item in enumerate(data["news_analysis"]):
    if i < 2:
        continue
    title = item.get("title", "")
    if title:
        clean = preprocess(title)
        vec = vectorizer.transform([clean])
        prob = model.predict_proba(vec)[0]
        score = round(prob[1] * 100, 2)
        item["confidence_score"] = score
        if "strategy_decision" in item:
            item["strategy_decision"]["confidence"] = score


with open("updated_confidence.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ Güncelleme tamamlandı: updated_confidence.json")

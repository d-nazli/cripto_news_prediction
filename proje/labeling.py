import pandas as pd

# 1. Temizlenmiş veri dosyasını yükle
try:
    df = pd.read_csv("cleaned_news.csv")
except FileNotFoundError:
    print("❗ 'cleaned_news.csv' bulunamadı. Önce 'cleaning.py' dosyasını çalıştırmalısın.")
    exit()

# 2. 'clean_text' sütunu var mı kontrol et
if 'clean_text' not in df.columns:
    print("❗ 'clean_text' sütunu bulunamadı. Dosya eksik ya da yanlış.")
    exit()

# 3. Boş temiz metinleri çıkar
df = df.dropna(subset=['clean_text'])

# 4. Etiketleme fonksiyonu (anahtar kelimeye göre)
def label_sentiment(text):
    # 📈 Olumlu (doğru/güven verici) haber anahtar kelimeleri
    positive_keywords = [
        'rise', 'bullish', 'surge', 'gain', 'increase', 'pump', 'partnership',
        'growth', 'boost', 'rally', 'support', 'adoption', 'breakout',
        'funding', 'innovation', 'approved', 'upgrade', 'recovery', 'uptrend',
        'milestone', 'record high', 'green candle', 'win', 'listed', 'integrated'
    ]

    # 📉 Olumsuz (yanıltıcı/kötü) haber anahtar kelimeleri
    negative_keywords = [
        'fall', 'bearish', 'crash', 'drop', 'decline', 'plummet', 'hacked',
        'lawsuit', 'ban', 'penalty', 'fraud', 'scam', 'arrested', 'breach',
        'exploit', 'loss', 'sell-off', 'dump', 'volatile', 'suspended',
        'collapse', 'fine', 'fined', 'negative', 'rejected', 'warning', 'probe',
        'regulation', 'shutdown', 'vulnerability', 'liquidation'
    ]

    text = str(text).lower()
    if any(word in text for word in positive_keywords):
        return 1  # olumlu/doğru haber
    elif any(word in text for word in negative_keywords):
        return 0  # olumsuz/yanıltıcı haber
    else:
        return None  # belirsiz → eğitimde kullanılmaz

# 5. Etiketleri uygula
df['label'] = df['clean_text'].apply(label_sentiment)

# 6. Etiket istatistiklerini yazdır
toplam = len(df)
etiketli = df['label'].notna().sum()
belirsiz = toplam - etiketli

print(f"\n🔍 Toplam haber sayısı: {toplam}")
print(f"✅ Etiketlenmiş haber sayısı: {etiketli}")
print(f"❌ Belirsiz (None) etiket sayısı: {belirsiz}")

# 7. Sadece etiketli verileri bırak
df = df.dropna(subset=['label'])

if len(df) == 0:
    print("❗ Etiketlenmiş haber kalmadı. Lütfen anahtar kelimeleri gözden geçir.")
    exit()

# 8. Yeni dosyaya kaydet
df.to_csv("labeled_news.csv", index=False)
print("\n✅ Etiketlenmiş veri 'labeled_news.csv' dosyasına kaydedildi.")

# 9. İlk 5 satırı göster
print("\n📄 Etiketli örnekler:")
print(df[['clean_text', 'label']].head())

# 🔎 Pozitif haber örnekleri (label = 1)
print("\n📈 Örnek Pozitif (label=1) haberler:")
print(df[df['label'] == 1][['clean_text']].head(5))

# 🔎 Negatif haber örnekleri (label = 0)
print("\n📉 Örnek Negatif (label=0) haberler:")
print(df[df['label'] == 0][['clean_text']].head(5))

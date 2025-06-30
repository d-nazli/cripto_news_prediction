import pandas as pd

# Etiketli veriyi yükle
df = pd.read_csv("labeled_news.csv")

# Etiket dağılımını göster
print("\n📊 Etiket Sayıları:")
print(df['label'].value_counts())

# Oranlarını da yazdırmak istersen:
print("\n📈 Oranlar (%):")
print(df['label'].value_counts(normalize=True) * 100)

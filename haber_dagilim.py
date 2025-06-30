import pandas as pd


df = pd.read_csv("labeled_news.csv")


print("\n📊 Etiket Sayıları:")
print(df['label'].value_counts())


print("\n📈 Oranlar (%):")
print(df['label'].value_counts(normalize=True) * 100)

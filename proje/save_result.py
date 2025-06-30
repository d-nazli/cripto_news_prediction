import json
import os

# Coin adı → sembol eşleşmesi
COIN_SYMBOLS = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "BNB": "BNB",
    "Solana": "SOL",
    "XRP": "XRP",
    "Cardano": "ADA",
    "Dogecoin": "DOGE",
    "Toncoin": "TON",
    "Avalanche": "AVAX",
    "Shiba Inu": "SHIB",
    "Polkadot": "DOT",
    "TRON": "TRX",
    "Chainlink": "LINK",
    "Polygon": "MATIC",
    "Litecoin": "LTC",
    "Uniswap": "UNI",
    "Bitcoin Cash": "BCH",
    "NEAR": "NEAR",
    "Internet Computer": "ICP",
    "Stellar": "XLM",
    "Aptos": "APT",
    "Ethereum Classic": "ETC",
    "VeChain": "VET",
    "Filecoin": "FIL",
    "Maker": "MKR",
    "Kaspa": "KAS",
    "Render": "RNDR",
    "Hedera": "HBAR",
    "Arbitrum": "ARB",
    "Optimism": "OP",
    "Fantom": "FTM",
    "Tezos": "XTZ",
    "Synthetix": "SNX",
    "Algorand": "ALGO",
    "EOS": "EOS",
    "Aave": "AAVE",
    "Lido DAO": "LDO",
    "Curve DAO Token": "CRV",
    "Theta": "THETA",
    "Flow": "FLOW",
    "Chiliz": "CHZ",
    "GALA": "GALA",
    "Stacks": "STX",
    "Immutable": "IMX",
    "The Sandbox": "SAND",
    "Decentraland": "MANA",
    "Enjin": "ENJ",
    "Ocean Protocol": "OCEAN",
    "Zilliqa": "ZIL"
}

# Başlıktan coin kısaltmasını tespit et
def detect_coin(title):
    for name, symbol in COIN_SYMBOLS.items():
        if name.lower() in title.lower():
            return symbol
    return "UNKNOWN"

# JSON’a haber kaydeden fonksiyon
def save_result(title, confidence_score, affected_coin, prediction, filepath="results.json"):
    new_entry = {
        "title": title,
        "confidence_score": confidence_score,
        "affected_coin": affected_coin,
        "prediction": prediction  # "up" veya "down"
    }

    # results.json varsa oku, yoksa boş liste oluştur
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError:
                data = []
    else:
        data = []

    # Aynı başlık varsa ekleme
    if any(entry["title"] == title for entry in data):
        print("⚠️ Bu haber zaten kayıtlı.")
        return

    # Yeni girdiyi ekle ve kaydet
    data.append(new_entry)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ Haber başarıyla results.json dosyasına kaydedildi.")

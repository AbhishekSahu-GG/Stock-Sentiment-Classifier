import pandas as pd

def process_data():
    rows = []
    with open("data/raw/Sentences_50Agree.txt", "r", encoding="ISO-8859-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentence, label = line.rsplit("@", 1)
            rows.append({"sentence": sentence, "sentiment_label": label})

    df = pd.DataFrame(rows)
    df.to_csv("data/raw/financial_phrasebank.csv", index=False)

    print(f"Loaded {len(df)} samples")
    print(df['sentiment_label'].value_counts())
    print(df.head(5).to_string())

if __name__ == "__main__":
    process_data()
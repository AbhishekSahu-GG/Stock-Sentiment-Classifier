from sklearn.model_selection import train_test_split
import pandas as pd
import os

def split_data():
    df = pd.read_csv("data/raw/financial_phrasebank_clean.csv")

    # First split: 80% train, 20% temp (becomes val + test)
    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df['sentiment_label'], random_state=42
    )

    # Second split: 50/50 split of temp into val and test (10% each of original)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['sentiment_label'], random_state=42
    )

    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    print("\nTrain distribution:\n", train_df['sentiment_label'].value_counts(normalize=True))
    print("\nVal distribution:\n", val_df['sentiment_label'].value_counts(normalize=True))
    print("\nTest distribution:\n", test_df['sentiment_label'].value_counts(normalize=True))

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

if __name__ == "__main__":
    split_data()
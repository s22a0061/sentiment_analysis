import pandas as pd
import os

# Define file paths based on your GitHub structure
amazon_path = "data/7817_1.csv"
twitter_path = "data/twitter_sentiment_small.csv"

def preview_dataset(path, name, text_col, label_col):
    if os.path.exists(path):
        print(f"\n--- {name} Dataset Preview ---")
        df = pd.read_csv(path)
        # Displaying first 5 rows of relevant columns
        print(df[[text_col, label_col]].head())
        print(f"Total Rows: {len(df)}")
    else:
        print(f"Error: {path} not found. Make sure you are in the correct directory.")

# Preview Amazon
preview_dataset(amazon_path, "Amazon", "reviews.text", "reviews.rating")

# Preview Twitter
preview_dataset(twitter_path, "Twitter", "text", "target")

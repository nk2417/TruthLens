import pandas as pd
import re
import kagglehub
import os
import contractions
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def load_kaggle():
    # Download dataset and read
    path = kagglehub.dataset_download("emineyetm/fake-news-detection-datasets")
    subdir = os.path.join(path, "News _dataset")
    
    fake_df = pd.read_csv(os.path.join(subdir, "Fake.csv"))
    true_df = pd.read_csv(os.path.join(subdir, "True.csv"))
    
    # Add labels
    fake_df["label"] = 0   # 0 = Fake
    true_df["label"] = 1   # 1 = True
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    return df.copy()

def combine_dataset(
    fake_path: str = "datasets/politifact/politifact_fake.csv",
    real_path: str = "datasets/politifact/politifact_real.csv",
    out_path: str = "datasets/processed/politifact_combined.csv",
):
    """
    Combine datasets into one CSV
    with a binary label column: 0 = fake, 1 = real.
    Any existing columns are preserved.
    """

    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake CSV not found at: {fake_path}")
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Real CSV not found at: {real_path}")

    # Load CSVs
    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    # Add / override label column
    fake_df["label"] = 0   # 0 = fake
    real_df["label"] = 1   # 1 = real

    # Combine
    combined = pd.concat([fake_df, real_df], ignore_index=True)

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"Saved combined dataset to: {out_path}")
    print("Shape:", combined.shape)

def append_text_column(csv_path, dataset_type):
    """
    Append 'text' column to CSV by reading from corresponding JSON files.
    
    Args:
        csv_path: Path to the CSV file
        dataset_type: Either 'fake' or 'real'
    """
    # Read the CSV
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Get the base directory for datasets
    base_dir = Path(csv_path).parent
    json_base_dir = base_dir / dataset_type
    
    # List to store text values
    texts = []
    missing_count = 0
    
    # Process each row
    for idx, row in df.iterrows():
        news_id = row['id']
        json_path = json_base_dir / news_id / "news content.json"
        
        try:
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    text = json_data.get('text', '')
                    texts.append(text)
            else:
                print(f"Warning: JSON file not found for {news_id}: {json_path}")
                texts.append('')
                missing_count += 1
        except Exception as e:
            print(f"Error reading JSON for {news_id}: {e}")
            texts.append('')
            missing_count += 1
    
    # Add the text column
    df['text'] = texts
    
    # Save the updated CSV
    output_path = csv_path
    df.to_csv(output_path, index=False)
    print(f"Updated {csv_path} with 'text' column")
    print(f"Total rows: {len(df)}, Missing text: {missing_count}")
    print()

if __name__ == "__main__":
    # Get the datasets directory
    script_dir = Path(__file__).parent
    datasets_dir = script_dir / "datasets"
    
    # Process fake CSV
    fake_csv = datasets_dir / "politifact_fake.csv"
    if fake_csv.exists():
        append_text_column(fake_csv, "fake")
    else:
        print(f"Warning: {fake_csv} not found")
    
    # Process real CSV
    real_csv = datasets_dir / "politifact_real.csv"
    if real_csv.exists():
        append_text_column(real_csv, "real")
    else:
        print(f"Warning: {real_csv} not found")
    
    print("Done!")

def clean_text(text):
    text = text.lower()  # lowercase everything
    text = contractions.fix(text)  # expand contractions (ie couldn't --> could not)
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
    return text

def clean_dataset(df):
    df = df.copy()
    # remove duplicates and empty
    df = df.drop_duplicates(subset=["text"])
    df = df.dropna(subset=["title", "text"])

     # cast to string just in case
    df["title"] = df["title"].astype(str)
    df["text"]  = df["text"].astype(str)
    
    # drop empty / whitespace-only title/text
    df = df[df["title"].str.strip() != ""]
    df = df[df["text"].str.strip() != ""]

    #Clean title and text
    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def train_logreg(X_train, y_train, max_features=5000, ngram_range=(1, 2), max_iter=1000):
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

def train_svm(X_train, y_train, max_features=5000, ngram_range=(1, 2), max_iter=1000, C=1.0):
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train model
    model = LinearSVC(max_iter=max_iter, C=C, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer
    

def train_nb(X_train, y_train, max_features=5000, ngram_range=(1, 2), alpha=1.0):
    # train a TF-IDF and nb baseline classifier
    # vectorize text
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # train model
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer


def eval_on_dataset(model, vectorizer, X_test, y_test):
    # Vectorize test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return accuracy, report

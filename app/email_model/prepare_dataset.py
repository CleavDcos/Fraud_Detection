import os
import pandas as pd
from bs4 import BeautifulSoup
import re

# ========== CLEAN TEXT ==========
def clean_text(text):
    if not text:
        return ""

    # Remove HTML
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Lowercase
    text = text.lower()

    return text


# ========== EXTRACT EMAIL ==========
def extract_email_text(file_path):
    try:
        with open(file_path, "r", encoding="latin-1") as f:
            content = f.read()

        # Extract subject
        subject_match = re.search(r"Subject:(.*)", content)
        subject = subject_match.group(1).strip() if subject_match else ""

        # Extract body
        parts = content.split("\n\n", 1)
        body = parts[1] if len(parts) > 1 else ""

        return clean_text(subject + " " + body)

    except:
        return ""


# ========== LOAD SPAMASSASSIN ==========
def load_spamassassin_data(base_path):
    data = []

    folders = {
        "spam": 1,
        "easy_ham": 0,
        "hard_ham": 0
    }

    for folder_name, label in folders.items():
        folder_path = os.path.join(base_path, folder_name)

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            text = extract_email_text(file_path)

            if text.strip():
                data.append({"text": text, "label": label})

    return pd.DataFrame(data)


# ========== LOAD PHISHING CSV ==========
def load_phishing_data(csv_path):
    df = pd.read_csv(csv_path)

    possible_cols = ["text", "email", "body", "content", "text_combined"]

    text_col = None
    for col in possible_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise Exception(f"No valid text column found. Columns: {df.columns}")

    df = df[[text_col]].copy()
    df.columns = ["text"]

    df["text"] = df["text"].astype(str).apply(clean_text)
    df["label"] = 1

    return df


#  MAIN 
def main():
    base_dir = os.path.dirname(__file__)

    spamassassin_path = os.path.join(base_dir, "data/raw/spamassassin")
    phishing_csv_path = os.path.join(base_dir, "data/raw/phishing/phishing_emails.csv")
    output_path = os.path.join(base_dir, "data/processed/emails.csv")

    print("Loading SpamAssassin dataset...")
    df_spam = load_spamassassin_data(spamassassin_path)

    print("Loading phishing dataset...")
    df_phish = load_phishing_data(phishing_csv_path)

    print("Combining datasets...")
    df = pd.concat([df_spam, df_phish], ignore_index=True)

    print("Shuffling dataset...")
    df = df.sample(frac=1).reset_index(drop=True)

    print("Saving dataset...")
    df.to_csv(output_path, index=False)

    print("Dataset ready!")
    print("Total samples:", len(df))
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
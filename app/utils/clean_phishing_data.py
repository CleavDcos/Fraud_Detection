import pandas as pd

# Load raw dataset
df = pd.read_csv("data/verified_online.csv")

# Keep only 'url' column
df = df[['url']]

# Remove duplicates
df = df.drop_duplicates()

# Remove empty rows
df = df.dropna()

# Save cleaned file (FIXED PATH)
df.to_csv("data/phishing_urls.csv", index=False)

print("Phishing dataset cleaned and saved!")
print("Total URLs:", len(df))
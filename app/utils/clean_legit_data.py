import pandas as pd

# Load dataset WITHOUT headers
df = pd.read_csv("data/tranco.csv", header=None)

# Assign column names
df.columns = ['rank', 'domain']

# Convert to URLs
df['url'] = "https://" + df['domain']

# Keep only url column
df = df[['url']]

# Remove duplicates
df = df.drop_duplicates()

# Remove empty rows
df = df.dropna()

# Save cleaned file
df.to_csv("data/legit_urls.csv", index=False)

print("Legit dataset cleaned and saved!")
print("Total URLs:", len(df))
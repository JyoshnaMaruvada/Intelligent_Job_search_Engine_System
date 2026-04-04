import pandas as pd
import re

# Load dataset
file_path = "/Users/jyoshnamaruvada/Downloads/Jyo/jobs_dataset.csv"
df = pd.read_csv(file_path)

# -----------------------------
# CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = str(text).lower()                      # lowercase
    text = re.sub(r'[^a-z\s]', '', text)          # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra spaces
    return text


# -----------------------------
# APPLY CLEANING
# -----------------------------
df['clean_title'] = df['jobtitle'].apply(clean_text)
df['clean_desc'] = df['description'].apply(clean_text)

# -----------------------------
# SAVE CLEANED DATA
# -----------------------------
output_path = "/Users/jyoshnamaruvada/Desktop/MyProjects/processed_jobs.csv"
df.to_csv(output_path, index=False)

print("✅ processed_jobs.csv created successfully!")
print(df[['clean_title', 'clean_desc']].head())
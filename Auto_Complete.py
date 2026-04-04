import pandas as pd
from collections import Counter

# Load processed dataset
file_path = "/Users/jyoshnamaruvada/Desktop/MyProjects/processed_jobs.csv"
df = pd.read_csv(file_path)

# -----------------------------
# BUILD QUERY LIST
# -----------------------------
# Use job titles as suggestions
titles = df['clean_title'].dropna().tolist()

# Count frequency of titles
title_freq = Counter(titles)

# -----------------------------
# AUTO-COMPLETE FUNCTION
# -----------------------------
def autocomplete(prefix, top_n=5):
    prefix = prefix.lower()

    # Find matching titles
    matches = [title for title in title_freq if title.startswith(prefix)]

    # Sort by frequency
    matches = sorted(matches, key=lambda x: title_freq[x], reverse=True)

    return matches[:top_n]

# -----------------------------
# TEST
# -----------------------------
queries = ["da", "soft", "data s", "eng"]

for q in queries:
    print(f"\nSuggestions for '{q}':")
    print(autocomplete(q))
import pandas as pd
from collections import Counter
from rapidfuzz import process, fuzz

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("processed_jobs.csv")

titles = df['clean_title'].dropna().str.lower().tolist()
companies = df['company'].dropna().str.lower().tolist()

# 🔥 Combine both
combined_list = titles + companies

title_freq = Counter(combined_list)

# =========================
# AUTOCOMPLETE
# =========================
def autocomplete(query, top_n=5):
    query = query.lower()

    matches = process.extract(
        query,
        title_freq.keys(),
        scorer=fuzz.token_sort_ratio,
        limit=top_n
    )

    suggestions = [match[0] for match in matches if match[1] > 60]

    return suggestions

# =========================
# CORRECT QUERY
# =========================
def correct_query(query):
    query = query.lower()

    best_match = process.extractOne(
        query,
        title_freq.keys(),
        scorer=fuzz.token_sort_ratio
    )

    if best_match and best_match[1] > 60:
        return best_match[0]

    return query
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load processed dataset
file_path = "/Users/jyoshnamaruvada/Desktop/MyProjects/processed_jobs.csv"
df = pd.read_csv(file_path)

# -----------------------------
# COMBINE TEXT (important)
# -----------------------------
df['combined_text'] = df['clean_title'] + " " + df['clean_desc']

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search_jobs(query, top_n=5):
    query = query.lower()

    # Transform query
    query_vec = vectorizer.transform([query])

    # Compute similarity
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get top results
    top_indices = similarity.argsort()[-top_n:][::-1]

    results = df.iloc[top_indices][['company', 'jobtitle', 'location']]

    return results

# -----------------------------
# TEST
# -----------------------------
queries = [
    "data scientist python",
    "software engineer java",
    "machine learning"
]

for q in queries:
    print(f"\nResults for: {q}")
    print(search_jobs(q))
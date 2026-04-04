import pandas as pd
from collections import Counter

# Load processed dataset
file_path = "/Users/jyoshnamaruvada/Desktop/MyProjects/processed_jobs.csv"
df = pd.read_csv(file_path)

# Combine cleaned text
all_text = " ".join(df['clean_title'].astype(str)) + " " + " ".join(df['clean_desc'].astype(str))

# Tokenize
words = all_text.split()

# Build vocabulary (word frequency)
word_freq = Counter(words)

# Show results
print("Total unique words:", len(word_freq))

# Show top 20 words
print("\nTop 20 words:")
for word, freq in word_freq.most_common(20):
    print(word, ":", freq)

# Save vocabulary (optional but useful)
vocab_df = pd.DataFrame(word_freq.items(), columns=["word", "frequency"])
vocab_df.to_csv("/Users/jyoshnamaruvada/Desktop/MyProjects/vocabulary.csv", index=False)

print("\n✅ vocabulary.csv saved successfully!")
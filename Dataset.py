import pandas as pd

# Load dataset
file_path = "/Users/jyoshnamaruvada/Downloads/Jyo/jobs_dataset.csv"
df = pd.read_csv(file_path)

# Preview
print(df.head())

# Check columns
print(df.columns)
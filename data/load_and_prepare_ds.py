from datasets import load_dataset
import pandas as pd


data = load_dataset("osyvokon/pavlick-formality-scores")

# Separate the data into train and test and add "formal" column for binary classification approaches
df_train = pd.DataFrame(data["train"])
df_train["formal"] = df_train["avg_score"].apply(lambda x: 0 if x <= 0.0 else 1)
df_test = pd.DataFrame(data["test"])
df_test["formal"] = df_test["avg_score"].apply(lambda x: 0 if x <= 0.0 else 1)

df_train.to_csv("data/train.csv", encoding="utf-8", sep=",", index=False)
df_test.to_csv("data/test.csv", encoding="utf-8", sep=",", index=False)
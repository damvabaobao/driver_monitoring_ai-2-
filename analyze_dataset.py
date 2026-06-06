import pandas as pd

df = pd.read_csv("driver_dataset.csv")

print(df.describe())
print()
print(df.info())
print()
print(df["Label"].value_counts())
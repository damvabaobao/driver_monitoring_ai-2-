import pandas as pd
df = pd.read_csv("driver_dataset.csv")
print(df[df["EAR"] <= 0])
print(df[df["MAR"] <= 0])
print(df[df["Pitch"].abs() > 90])
print(df[df["Yaw"].abs() > 90])

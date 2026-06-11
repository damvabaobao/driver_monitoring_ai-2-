import pandas as pd
df = pd.read_csv("driver_dataset.csv")
features = ["EAR", "MAR", "Pitch", "Yaw"]

for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        print(col)\
        
        print("Outliers:", len(outliers))
        print(outliers.head())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("driver_dataset.csv")

features = [
        "EAR",
        "MAR",
        "Pitch",
        "Yaw"
]

for feature in features:
        plt.figure(figsize=(8, 4))
        sns.boxplot(
                x = "Label",
                y = feature,
                data = df
        )
        plt.title(feature)
        plt.show()
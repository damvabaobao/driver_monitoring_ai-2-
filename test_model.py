import joblib
import pandas as pd

model = joblib.load("driver_drowsiness_rf.pkl")

sample = pd.DataFrame([[
        0.15, # EAR
        0.4, # MAR
        15.0, #Pitch
        55.0 # Yaw
]], columns = ["EAR", "MAR", "Pitch", "Yaw"])

prediction = model.predict(sample)
if prediction[0] == 0:
        print("AWAKE")
else:
        print("DROWSY")
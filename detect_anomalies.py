
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generate synthetic SCADA-like data
np.random.seed(42)
normal_data = np.random.normal(loc=100, scale=5, size=(100, 1))
anomalies = np.random.normal(loc=150, scale=1, size=(5, 1))
data = np.vstack((normal_data, anomalies))
df = pd.DataFrame(data, columns=["SensorValue"])

# Train Isolation Forest model
model = IsolationForest(contamination=0.05)
df["anomaly"] = model.fit_predict(df[["SensorValue"]])

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(df["SensorValue"], label="Sensor Value")
plt.scatter(df[df["anomaly"] == -1].index, df[df["anomaly"] == -1]["SensorValue"], color='red', label="Anomaly")
plt.legend()
plt.title("Anomaly Detection in SCADA-like Sensor Data")
plt.xlabel("Time Index")
plt.ylabel("Sensor Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("anomaly_plot.png")
plt.show()

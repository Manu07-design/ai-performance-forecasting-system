
# AI Performance Forecasting System


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 1. Load Dataset
df = pd.read_csv("performance_data.csv")

print("\n Dataset:")
print(df)


# 2. Create Lag Feature

df["lag_1"] = df["marks"].shift(1)
df.dropna(inplace=True)


# 3. Train Forecast Model

X = df[["lag_1"]]
y = df["marks"]

model = LinearRegression()
model.fit(X, y)


# 4. Predict Next Week

last_mark = df["marks"].iloc[-1]
prediction = model.predict([[last_mark]])

print(f"\n Predicted Next Week Marks: {prediction[0]:.2f}")


# 5. Visualize Trend

plt.plot(df["week"], df["marks"], label="Actual Marks")
plt.scatter(df["week"].iloc[-1] + 1, prediction, color="red", label="Forecast")

plt.title("Performance Forecast")
plt.xlabel("Week")
plt.ylabel("Marks")
plt.legend()
plt.show()

print("\n Forecasting system executed successfully.")
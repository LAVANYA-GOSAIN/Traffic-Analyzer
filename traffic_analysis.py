# # ==========================================
# # Traffic Congestion Analysis (Terminal-Based)
# # Analyse how private vehicle density affects traffic speed
# # ==========================================

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_absolute_error

# # -------------------------------
# # 1. LOAD DATASET
# # -------------------------------
# # Change filename if needed
# df = pd.read_csv("/Users/lavanyagosain/Desktop/Skilldevelopment/28:01:26/traffic_data.csv")

# # print("\nDataset loaded successfully!")
# # print(df.head())

# # -------------------------------
# # 2. SELECT IMPORTANT PARAMETERS
# # -------------------------------
# features = [
#     "density_veh_per_km",
#     "occupancy_pct",
#     "avg_wait_time_s"
# ]

# target = "avg_speed_kmph"

# df = df[features + [target]].dropna()

# # -------------------------------
# # 3. EXPLORATORY ANALYSIS (HEATMAP)
# # -------------------------------
# plt.figure(figsize=(8, 6))
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()

# # -------------------------------
# # 3A. DENSITY vs SPEED SCATTER PLOT
# # -------------------------------
# plt.figure(figsize=(7, 5))
# plt.scatter(df["density_veh_per_km"], df["avg_speed_kmph"], alpha=0.5)
# plt.xlabel("Vehicle Density (veh/km)")
# plt.ylabel("Average Speed (km/h)")
# plt.title("Vehicle Density vs Traffic Speed")
# plt.show()

# # -------------------------------
# # 4. TRAIN–TEST SPLIT
# # -------------------------------
# X = df[features]
# y = df[target]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # -------------------------------
# # 5. TRAIN REGRESSION MODEL
# # -------------------------------
# model = LinearRegression()
# model.fit(X_train, y_train)

# # print("\nModel training completed!")

# # -------------------------------
# # 6. MODEL EVALUATION
# # -------------------------------
# y_pred = model.predict(X_test)

# print("\nModel Evaluation:")
# print("R2 Score:", r2_score(y_test, y_pred))
# print("MAE:", mean_absolute_error(y_test, y_pred))

# # -------------------------------
# # 7. ACTUAL vs PREDICTED GRAPH
# # -------------------------------
# plt.figure(figsize=(7, 5))
# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual Speed (km/h)")
# plt.ylabel("Predicted Speed (km/h)")
# plt.title("Actual vs Predicted Traffic Speed")
# plt.show()

# # -------------------------------
# # 8. TERMINAL INPUT FOR PREDICTION
# # -------------------------------
# print("\n--- Traffic Speed Prediction ---")

# density = float(input("Enter vehicle density (veh/km): "))
# occupancy = float(input("Enter road occupancy (%): "))
# wait_time = float(input("Enter average waiting time (sec): "))

# input_data = pd.DataFrame(
#     [[density, occupancy, wait_time]],
#     columns=features
# )

# predicted_speed = model.predict(input_data)[0]

# print(f"\nPredicted Average Traffic Speed: {predicted_speed:.2f} km/h")

# # Optional interpretation
# if predicted_speed < 20:
#     print("Traffic Condition: Heavy Congestion")
# elif predicted_speed < 40:
#     print("Traffic Condition: Moderate Congestion")
# else:
#     print("Traffic Condition: Free Flow")

# print("\nTraffic congestion analysis completed successfully.")




# ==========================================
# Traffic Congestion Analysis Using ML
# Comparative Study of 6 ML Models
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
# 1. LOAD DATASET
# -------------------------------
df = pd.read_csv("/Users/lavanyagosain/Desktop/Skilldevelopment/28:01:26/traffic_data.csv")

print("\nDataset Loaded Successfully!")

# -------------------------------
# 2. SELECT IMPORTANT FEATURES
# -------------------------------
features = [
    "density_veh_per_km",
    "occupancy_pct",
    "avg_wait_time_s"
]

target = "avg_speed_kmph"

df = df[features + [target]].dropna()

# -------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# -------------------------------

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Density vs Speed (Line Graph)
plt.figure(figsize=(7, 5))
sorted_df = df.sort_values("density_veh_per_km")
plt.plot(sorted_df["density_veh_per_km"],
         sorted_df["avg_speed_kmph"])
plt.xlabel("Vehicle Density (veh/km)")
plt.ylabel("Average Speed (km/h)")
plt.title("Vehicle Density vs Traffic Speed (Trend)")
plt.show()

# -------------------------------
# 4. TRAIN–TEST SPLIT
# -------------------------------
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. TRAIN MULTIPLE ML MODELS
# -------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(max_iter=10000),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Regressor": SVR(kernel='rbf')
}

results = {}

print("\nModel Performance Comparison:")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {"R2": r2, "MAE": mae}

    print(f"\n{name}")
    print("R2 Score:", round(r2, 4))
    print("MAE:", round(mae, 4))

# -------------------------------
# 6. BEST MODEL SELECTION
# -------------------------------
best_model_name = max(results, key=lambda x: results[x]["R2"])
final_model = models[best_model_name]
final_model.fit(X_train, y_train)

print("\n---------------------------------")
print("Best Performing Model:", best_model_name)
print("Best R2 Score:", round(results[best_model_name]["R2"], 4))
print("---------------------------------")

# -------------------------------
# 7. MODEL COMPARISON GRAPH
# -------------------------------
model_names = list(results.keys())
r2_scores = [results[m]["R2"] for m in model_names]

plt.figure(figsize=(8, 5))
plt.bar(model_names, r2_scores)
plt.xticks(rotation=45)
plt.ylabel("R2 Score")
plt.title("Model Comparison (R2 Score)")
plt.show()

# -------------------------------
# 8. CONGESTION CLASSIFICATION
# -------------------------------
final_predictions = final_model.predict(X_test)

def congestion_label(speed):
    if speed < 20:
        return "Heavy"
    elif speed < 40:
        return "Moderate"
    else:
        return "Free Flow"

congestion_levels = [congestion_label(s) for s in final_predictions]

# Congestion Distribution Graph
congestion_counts = pd.Series(congestion_levels).value_counts()

plt.figure(figsize=(6, 5))
plt.bar(congestion_counts.index, congestion_counts.values)
plt.xlabel("Traffic Condition")
plt.ylabel("Number of Occurrences")
plt.title("Predicted Traffic Congestion Distribution")
plt.show()

# -------------------------------
# 9. TERMINAL PREDICTION
# -------------------------------
print("\n--- Traffic Speed Prediction ---")

density = float(input("Enter vehicle density (veh/km): "))
occupancy = float(input("Enter road occupancy (%): "))
wait_time = float(input("Enter average waiting time (sec): "))

input_data = pd.DataFrame(
    [[density, occupancy, wait_time]],
    columns=features
)

predicted_speed = final_model.predict(input_data)[0]

print(f"\nPredicted Average Traffic Speed: {predicted_speed:.2f} km/h")

# Congestion Interpretation
if predicted_speed < 20:
    print("Traffic Condition: Heavy Congestion")
elif predicted_speed < 40:
    print("Traffic Condition: Moderate Congestion")
else:
    print("Traffic Condition: Free Flow")

print("\nTraffic congestion analysis completed successfully.")









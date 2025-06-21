import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('synthetic_wtp_laptop_data.csv')
except FileNotFoundError:
    print("Error: 'synthetic_wtp_laptop_data.csv' not found.")
    print("Please make sure the CSV file is in the same directory as the script.")
    exit()

# Define features (X) and target (y)
features = ['Memory', 'Storage', 'CPU_class', 'Screen_size', 'year']
X = df[features]
y = df['price']

# Initialize and train the Random Forest Regressor model
# We use the whole dataset for training to get the most accurate model based on the provided data
model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
model.fit(X, y)

print(f"Model OOB score: {model.oob_score_:.4f}")
print("-" * 30)

# --- Feature Permutation Importance ---
print("Calculating feature importance using permutation...")
perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = perm_importance.importances_mean.argsort()

for i in sorted_idx:
    print(f"{features[i]:<12}: {perm_importance.importances_mean[i]:.3f} +/- {perm_importance.importances_std[i]:.3f}")
print("-" * 30)


# --- Analysis for Specific Upgrades ---

# Base product specification
base_spec = {
    'Memory': 16,
    'Storage': 512,
    'CPU_class': 1,
    'Screen_size': 14.0,
    'year': 2025
}
base_spec_df = pd.DataFrame([base_spec])
base_price = model.predict(base_spec_df[features])[0]

print("Current Model Specs and Selling Price")
for key, value in base_spec.items():
    print(f"{key:<12}: {value}")
print(f"Predicted Selling Price: {base_price:,.0f} yen")
print("-" * 30)


# Define potential upgrades
upgrades = {
    "Add 16 GB memory": {"Memory": 32},
    "Add 512 GB storage": {"Storage": 1024},
    "Upgrade CPU class by 1 level": {"CPU_class": 2},
    "Increase screen to 16 inches": {"Screen_size": 16.0}
}

results = {}

print("Analyzing potential upgrades...")
for upgrade_name, upgrade_spec in upgrades.items():
    # Create new spec with the upgrade
    new_spec_dict = base_spec.copy()
    new_spec_dict.update(upgrade_spec)
    
    # Predict new price
    new_spec_df = pd.DataFrame([new_spec_dict])
    new_price = model.predict(new_spec_df[features])[0]
    
    price_increase = new_price - base_price
    results[upgrade_name] = price_increase
    
    print(f"Upgrade: {upgrade_name.title()}")
    print(f"New Spec: {upgrade_spec}")
    print(f"Predicted New Price: {new_price:,.0f} yen")
    print(f"Price Increase: {price_increase:,.0f} yen\n")

# --- Conclusion ---
best_upgrades = sorted(results.items(), key=lambda item: item[1], reverse=True)

print("-" * 30)
print("Conclusion:")
print("Based on the model, the upgrades that will yield the most results (highest price increase) are:")
for i in range(2):
    upgrade_name, increase = best_upgrades[i]
    print(f"{i+1}. {upgrade_name.title()}: Increase of ~{increase:,.0f} yen")

print("\nFull ranking of upgrades by price increase:")
for i, (upgrade_name, increase) in enumerate(best_upgrades):
     print(f"{i+1}. {upgrade_name.title()}: ~{increase:,.0f} yen") 
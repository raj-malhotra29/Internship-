import numpy as np
import pandas as pd

# Set seed for reproducible student simulation
np.random.seed(42)

# 1. Simulate Logistics Data Set (500 records)
num_shipments = 500

data = {
    "Shipment_ID": [f"SHIP_{1000 + i}" for i in range(num_shipments)],
    "Origin": np.random.choice(
        ["Delhi", "Mumbai", "Bangalore", "Lucknow", "Kolkata"], num_shipments
    ),
    "Destination": np.random.choice(
        ["Chennai", "Hyderabad", "Jaipur", "Pune", "Ahmedabad"], num_shipments
    ),
    "Transport_Mode": np.random.choice(
        ["Road", "Rail", "Air"], num_shipments, p=[0.6, 0.3, 0.1]
    ),
    "Distance_KM": np.random.randint(150, 1600, size=num_shipments),
    "Shipping_Cost_USD": np.random.uniform(20.0, 350.0, size=num_shipments).round(
        2
    ),
    "Estimated_Days": np.random.randint(1, 6, size=num_shipments),
    "Actual_Days": np.random.randint(1, 9, size=num_shipments),
}

df = pd.DataFrame(data)

# 2. Feature Engineering & KPIs
df["Delay_Days"] = df["Actual_Days"] - df["Estimated_Days"]
df["Is_On_Time"] = df["Delay_Days"].apply(lambda x: 1 if x <= 0 else 0)

total_orders = len(df)
otd_rate = (df["Is_On_Time"].sum() / total_orders) * 100
avg_cost = df["Shipping_Cost_USD"].mean()
avg_delay_hrs = df[df["Delay_Days"] > 0]["Delay_Days"].mean() * 24

# 3. Output Analytical Summary
print("==================================================")
print("  LOGISTICS DATA ANALYTICS: WEEK 1 EXPLORATION   ")
print("==================================================")
print(f"Total Shipments Evaluated : {total_orders}")
print(f"On-Time Delivery Rate     : {otd_rate:.2f}%")
print(f"Average Shipping Cost     : ${avg_cost:.2f}")
print(f"Average Late Delay        : {avg_delay_hrs:.1f} Hours")
print("==================================================\n")

# Grouped breakdown by Transport Mode
mode_summary = df.groupby("Transport_Mode").agg(
    Total_Shipments=("Shipment_ID", "count"),
    Avg_Cost=("Shipping_Cost_USD", "mean"),
    On_Time_Rate=("Is_On_Time", "mean"),
)

print("Performance Breakdown by Transport Mode:")
print(mode_summary)
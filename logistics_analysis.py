import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. GENERATE SYNTHETIC LOGISTICS DATASET
# ==========================================
n_samples = 500

data = {
    'Shipment_ID': [f'SHP_{1000 + i}' for i in range(n_samples)],
    'Origin_City': np.random.choice(
        ['Delhi', 'Mumbai', 'Bangalore', 'Lucknow', 'Kolkata'], size=n_samples
    ),
    'Destination_City': np.random.choice(
        ['Chennai', 'Hyderabad', 'Ahmedabad', 'Pune', 'Jaipur'], size=n_samples
    ),
    'Transport_Mode': np.random.choice(
        ['Road', 'Rail', 'Air'], size=n_samples, p=[0.6, 0.3, 0.1]
    ),
    'Shipment_Weight_KG': np.round(
        np.random.uniform(10.0, 500.0, size=n_samples), 2
    ),
    'Distance_KM': np.random.randint(150, 2500, size=n_samples),
    'Promised_Delivery_Days': np.random.randint(2, 7, size=n_samples),
}

df = pd.DataFrame(data)

# Derive calculated metrics
mode_speed_factor = {'Road': 1.0, 'Rail': 0.85, 'Air': 0.4}
df['Delivery_Time_Days'] = np.round(
    (df['Distance_KM'] / 350) * df['Transport_Mode'].map(mode_speed_factor)
    + np.random.normal(0.5, 0.8, n_samples),
    1,
)
df['Delivery_Time_Days'] = df['Delivery_Time_Days'].clip(lower=1.0)

df['Delay_Status'] = np.where(
    df['Delivery_Time_Days'] > df['Promised_Delivery_Days'],
    'Delayed',
    'On-Time',
)

mode_cost_rate = {'Road': 12, 'Rail': 8, 'Air': 35}
df['Transportation_Cost_INR'] = np.round(
    150
    + (df['Shipment_Weight_KG'] * 0.5)
    + (
        df['Distance_KM']
        * df['Transport_Mode'].map(mode_cost_rate)
        * 0.05
    )
    + np.random.normal(0, 100, n_samples),
    2,
)

# Save generated dataset
csv_path = 'logistics_data.csv'
df.to_csv(csv_path, index=False)
print(f'Dataset successfully saved to {csv_path}')

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print('\n--- DESCRIPTIVE STATISTICS ---')
numeric_cols = [
    'Shipment_Weight_KG',
    'Distance_KM',
    'Delivery_Time_Days',
    'Transportation_Cost_INR',
]
print(df[numeric_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])

# ==========================================
# 3. GENERATE VISUALIZATIONS
# ==========================================
plt.style.use(
    'seaborn-v0_8-whitegrid'
    if 'seaborn-v0_8-whitegrid' in plt.style.available
    else 'default'
)

# Chart 1: Histogram - Delivery Time Distribution
plt.figure(figsize=(8, 4.5), dpi=300)
sns.histplot(df['Delivery_Time_Days'], kde=True, color='#1f77b4', bins=20)
plt.title(
    'Distribution of Delivery Times (Days)', fontsize=12, fontweight='bold'
)
plt.xlabel('Delivery Time (Days)', fontsize=10)
plt.ylabel('Shipment Frequency', fontsize=10)
plt.tight_layout()
plt.savefig('chart1_histogram.png')
plt.close()

# Chart 2: Scatter Plot - Cost vs. Distance by Transport Mode
plt.figure(figsize=(8, 4.5), dpi=300)
sns.scatterplot(
    data=df,
    x='Distance_KM',
    y='Transportation_Cost_INR',
    hue='Transport_Mode',
    palette={'Road': '#2ca02c', 'Rail': '#ff7f0e', 'Air': '#d62728'},
    alpha=0.8,
    s=60,
)
plt.title(
    'Transportation Cost vs. Distance by Transport Mode',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Distance (KM)', fontsize=10)
plt.ylabel('Transportation Cost (INR)', fontsize=10)
plt.legend(title='Transport Mode')
plt.tight_layout()
plt.savefig('chart2_scatterplot.png')
plt.close()

# Chart 3: Correlation Heatmap
plt.figure(figsize=(6.5, 4.5), dpi=300)
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='Blues',
    fmt='.2f',
    linewidths=0.5,
    cbar=True,
)
plt.title(
    'Correlation Matrix of Key Metrics', fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig('chart3_heatmap.png')
plt.close()

# Chart 4: Grouped Count Plot - Delay Status by Transport Mode
plt.figure(figsize=(8, 4.5), dpi=300)
sns.countplot(
    data=df,
    x='Transport_Mode',
    hue='Delay_Status',
    palette={'On-Time': '#2ca02c', 'Delayed': '#d62728'},
)
plt.title(
    'Shipment Delay Status Across Transport Modes',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Transport Mode', fontsize=10)
plt.ylabel('Number of Shipments', fontsize=10)
plt.legend(title='Status')
plt.tight_layout()
plt.savefig('chart4_countplot.png')
plt.close()

print('\nAll 4 charts generated and saved successfully.')
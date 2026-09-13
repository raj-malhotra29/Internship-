import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def execute_logistics_pipeline(file_path: str) -> pd.DataFrame:
    # 1. Load Raw Data
    df = pd.read_csv(file_path)
    print(f"Initial Record Count: {len(df)}")

    # 2. Deduplication
    df = df.drop_duplicates(subset=["Shipment_ID"]).reset_index(drop=True)
    print(f"Record Count Post-Deduplication: {len(df)}")

    # 3. Handling Missing Values (Grouped Median Imputation)
    df["Actual_Transit_Days"] = df.groupby("Origin_Port")[
        "Actual_Transit_Days"
    ].transform(lambda x: x.fillna(x.median()))
    df["Shipping_Cost_USD"] = df["Shipping_Cost_USD"].fillna(
        df["Shipping_Cost_USD"].median()
    )
    df["Delivery_Status"] = df["Delivery_Status"].fillna(
        df["Delivery_Status"].mode()[0]
    )

    # 4. Statistical Outlier Capping via IQR
    def apply_iqr_capping(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.clip(series, lower_bound, upper_bound)

    df["Shipping_Cost_USD"] = apply_iqr_capping(df["Shipping_Cost_USD"])
    df["Package_Weight_KG"] = apply_iqr_capping(df["Package_Weight_KG"])

    # 5. Feature Engineering (Transit Delay Flag)
    df["Delay_Days"] = np.maximum(
        0, df["Actual_Transit_Days"] - df["Planned_Transit_Days"]
    )

    # 6. Min-Max Normalization
    scaler = MinMaxScaler()
    scaling_cols = [
        "Planned_Transit_Days",
        "Actual_Transit_Days",
        "Shipping_Cost_USD",
        "Package_Weight_KG",
        "Delay_Days",
    ]
    df[scaling_cols] = scaler.fit_transform(df[scaling_cols])

    return df


if __name__ == "__main__":
    cleaned_df = execute_logistics_pipeline("raw_logistics_data.csv")
    cleaned_df.to_csv("cleaned_logistics_data.csv", index=False)
    print("Logistics Data Preprocessing Complete.")
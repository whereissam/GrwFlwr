import numpy as np
from flwr.common import NDArrays
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression

import joblib


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(csv_filepath, partition_id, target_column="Water_Demand_Liters", test_size=0.2, random_state=42):

    # Read and split in X and y 
    df = pd.read_csv(csv_filepath)  
    df = df[df["Partition"] == partition_id]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split in train and test 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler



def train_model(X_train, y_train, scaler=None, model_path="model.joblib", scaler_path="scaler.joblib"):
   
    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    if scaler is not None:
        joblib.dump(scaler, scaler_path)


def evaluate_model(X_test, y_test, model_path="model.joblib", scaler_path="scaler.joblib"):
  
    model = joblib.load(model_path)
    # scaler = joblib.load(scaler_path)

    y_pred = model.predict(X_test)
    
    metrics = {
        "r2_score": float(r2_score(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred))
    }
    
    return metrics


def predict_water_demand(
    field_size_ha, 
    temperature_c, 
    rainfall_mm, 
    soil_humidity_percent, 
    model_path="model.joblib", 
    scaler_path="scaler.joblib"
):

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    input_data = pd.DataFrame(
        [[1, field_size_ha, temperature_c, rainfall_mm, soil_humidity_percent]],
        columns=["Partition", "Field_Size_ha", "Temperature_C", "Rainfall_mm", "Soil_Humidity_Percent"]
    )

    input_scaled = scaler.transform(input_data)

    predicted_demand = model.predict(input_scaled)

    return round(float(predicted_demand[0]), 2)


if __name__ == "__main__":

    # # Prepare data
    # X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data("../../data.csv", partition_id=1, target_column="Water_Demand_Liters", test_size=0.2, random_state=42)

    # # Train model
    # train_model(X_train_scaled, y_train, scaler=scaler, model_path="model.joblib", scaler_path="scaler.joblib")

    # # Evaluate model
    # metrics = evaluate_model(X_test_scaled, y_test, model_path="model.joblib", scaler_path="scaler.joblib")
    # print(f"R² Score: {metrics['r2_score']:.4f}")
    # print(f"MSE:      {metrics['mse']:,.2f}")

    # Inference
    # Beispiel-Aufruf:
    # Feld von 10.5 ha, 28°C Temperatur, 2.0 mm Regen, 25% Bodenfeuchte
    field_size = 10.5
    temp = 28.0
    rain = 2.0
    humidity = 25.0

    water_needed = predict_water_demand(field_size, temp, rain, humidity)
    print(f"Benötigte Wassermenge für das Feld: {water_needed:,.2f} Liter")
    






import csv
import random

def generate_csv_data(filename="data.csv", num_samples=1000, num_partitions=2):
   
    header = [
        "Partition",
        "Field_Size_ha",
        "Temperature_C",
        "Rainfall_mm",
        "Soil_Humidity_Percent",
        "Water_Demand_Liters"
    ]
    
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)  
        
        for i in range(num_samples):
            # Assign partition (e.g., evenly distributed across clients)
            partition = i % num_partitions
            
            # Generate features for irrigation prediction
            field_size = round(random.uniform(0.5, 20.0), 2)       # 0.5 to 20 hectares
            temperature = round(random.uniform(10.0, 38.0), 1)     # 10°C to 38°C
            rainfall = round(random.uniform(0.0, 25.0), 1)        # 0 to 25 mm
            soil_humidity = round(random.uniform(10.0, 60.0), 1)   # 10% to 60% humidity
            
            # Target formula with noise
            base_demand = (
                (field_size * 5000) 
                + (temperature * 300) 
                - (rainfall * 400) 
                - (soil_humidity * 250) 
                + 15000
            )
            
            # Add Gaussian noise
            noise = random.gauss(0, 2500)
            water_demand = max(0, round(base_demand + noise, 2))  # Ensure demand isn't negative
            
            # Write row
            writer.writerow([
                partition, 
                field_size, 
                temperature, 
                rainfall, 
                soil_humidity, 
                water_demand
            ])


if __name__ == "__main__":
    generate_csv_data()
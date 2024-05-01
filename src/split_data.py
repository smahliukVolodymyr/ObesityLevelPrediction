import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Шлях до файлу "variant_1.csv"
file_path = os.path.join("data", "ObesityDataSet.csv")

# Завантаження даних з файлу "variant_1.csv"
data = pd.read_csv(file_path)

# Розділення даних на нові вхідні дані та тестові дані у співвідношенні 10:90
new_input,train = train_test_split(data, test_size=0.8, random_state=42)

output_directory = "data"
new_input.to_csv(os.path.join(output_directory, "new_input.csv"), index=False)
train.to_csv(os.path.join(output_directory, "train.csv"), index=False)

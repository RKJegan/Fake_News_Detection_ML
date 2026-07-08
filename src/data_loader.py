import pandas as pd
import numpy as np
import os


def load_data():

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, 'data1')

    fake_path = os.path.join(data_path, 'Fake.csv')
    true_path = os.path.join(data_path, 'True.csv')

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake['label'] = 0
    true['label'] = 1

    data = pd.concat([fake, true], ignore_index=True)

    data = data.sample(frac=1,random_state=42).reset_index(drop=True)

    data['content'] = data['title'] + " " + data['text']  # Combine title and text into a single content column

    return data


if __name__ == "__main__":
    data = load_data()
    print(data.head())
    print("Data shape:", data.shape)
    print(data.info())
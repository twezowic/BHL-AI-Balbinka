import pandas as pd
import numpy as np

def get_labels(filepath: str):
    data = pd.read_csv(filepath)
    return data


def find_peakflux(data: pd.DataFrame, id):
    return data[data['id'] == id]["peak_flux"].iloc[0]


def normalize(data, id):
    return (np.log10(find_peakflux(data, id)) + 9).astype(float)

if __name__ == "__main__":
    a = get_labels("./data/SDOBenchmark-data-example/training/meta_data.csv")
    print(find_peakflux(a, '11388_2012_01_07_02_27_01_0'))

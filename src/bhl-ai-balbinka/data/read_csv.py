import pandas as pd


def get_labels(filepath: str):
    data = pd.read_csv(filepath)
    return data


def find_peakflux(data: pd.DataFrame, id):
    return data[data['id'] == id]["peak_flux"][0]



if __name__ == "__main__":
    a = get_labels("./data/SDOBenchmark-data-example/training/meta_data.csv")
    print(find_peakflux(a, '11389_2012_01_01_19_06_00_0'))

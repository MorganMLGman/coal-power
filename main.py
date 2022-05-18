from scipy.io import loadmat
import pandas as pd
from matplotlib import pyplot as plt

DATA_FILE = "a08r.mat"
ARRAY_NAME = "a08r"
SPS = 300 # Samples per second

if __name__ == "__main__":
    data = pd.DataFrame(loadmat(DATA_FILE)[ARRAY_NAME], columns=(["ch1", "ch2", "ch3", "ch4", "ch5"]))
    print(data.head(3))    
    
    plt.figure(figsize=(25, 15))
    plt.plot(data.head(300000).rolling(SPS).mean(), label=data.columns)
    plt.legend()
    plt.show()

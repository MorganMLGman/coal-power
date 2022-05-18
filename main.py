from scipy.io import loadmat
import pandas as pd

# %% MAIN
if __name__ == "__main__":
    data = pd.DataFrame(loadmat("a08r.mat")['a08r'], columns=(["channel1", "channel2", "channel3", "channel4", "channel5"]))
    print(data)
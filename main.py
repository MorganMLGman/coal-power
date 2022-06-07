# %%
import logging
from time import perf_counter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.signal import find_peaks, argrelextrema
import statsmodels.api as sm
import seaborn as sns
import threading as th
from tabulate import tabulate as tb

# %%
DATA_FILE = "a08r.mat"
ARRAY_NAME = "a08r"
SPS = 8192 # Samples per second

# %%

logger = logging.getLogger("projekt")
logger_stream = logging.StreamHandler()
logger.handlers.clear()
logger.addHandler(logger_stream)
logger.setLevel(logging.DEBUG)

# %%
def findMaximums(data: pd.DataFrame, column: str, width: int = 10, distance: int = SPS*45, threshold: list = None, prominence: float = 0.1) -> list:
    """Funkcja służąca do wyszukiwanie punktów maksimum w otrzymanych danych

    Args:
        data (pd.DataFrame): dataFrame z danymi, cały niezmodyfikowany
        column (str): nazwa kolumny w której chcemy szukać
        width (int, optional): minimalna szerokość (x) piku. Defaults to 10.
        distance (int, optional): minimlna odległość (x) pomiędzy szukanymi punktami. Defaults to SPS*45.
        threshold (list, optional): minimlna/maksymalna odległość (y) pomiędzy szukanymi punktami. Defaults to None.
        prominence (float, optional): eeee, to nie wiem, ale czasami jak większe to pomaga eliminować podwójne punkty. Defaults to 0.1.

    Returns:
        list: Zwraca listę z punktami maksimum
    """
    logger.debug("Function: findMaximums")
    ret = []
    ret = find_peaks(data[column], width=width, distance=distance, threshold=threshold, prominence=prominence)[0].tolist()
    return ret  

# %%
def dataSplit(data: pd.DataFrame, spliter: list, channel: str = "all"):
    """Funkcja służąca do podziału danych na podzbiory, według podanej listy podziału. Przykładowo dla punktów podziału 1, 2, 3, zwraca przedziały [1, 2], [2, 3]

    Args:
        data (pd.DataFrame): dataFrame z danymi, cały niezmodyfikowany
        spliter (list): lista, wykorzystywana do podziału
        channel (str): nazwa kanału, który ma zostać zpisany, `all` jeżeli wszystkie

    Returns:
        pd.DataFrame: Zwraca dataFrame z przedziałami
        list: list of keys to access buckets
    """
    logger.debug("Function: dataSplit")
    ret = pd.DataFrame()
    num_of_buckets = len(spliter) - 1
    keys = [f"bucket{i}" for i in range(num_of_buckets)]
    tmp = []
    
    if "all" == channel:
        logger.debug("Spliting all channels")
        for item in range(num_of_buckets):
            tmp.append(data[spliter[item]:(spliter[item + 1] - 1)])
        ret = pd.concat(tmp, keys=keys)
        
    elif channel in data.columns:
        logger.debug(f"Spliting {channel}")
        for item in range(num_of_buckets):
            tmp.append(data[channel][spliter[item]:(spliter[item + 1] - 1)])
        ret = pd.concat(tmp, keys=keys)
        
    else:
        logger.debug(f"Parameter channel is not valid")
        ret = None
        
    return ret, keys

# %%
def descriptiveStats(bucket: pd.Series) -> dict:
    logger.debug("Function: descriptiveStats")
    ret = dict.fromkeys(["mean", "median", "min", "min_idx" "max", "max_idx", "std", "var", "auto_corr", "roll_mean"])
    
    ret["mean"] = bucket.mean()
    ret["median"] = bucket.median()
    ret["min"] = bucket.min()
    ret["min_idx"] = bucket.idxmin()
    ret["max"] = bucket.max()
    ret["max_idx"] = bucket.idxmax()
    ret["std"] = bucket.std()
    ret["var"] = bucket.var()
    ret["auto_corr"] = pd.Series(sm.tsa.acf(bucket, nlags=bucket.size), index=range(bucket.first_valid_index(), bucket.last_valid_index() + 1))
    ret["roll_mean"] = bucket.rolling(SPS).mean()
        
    return ret

# %%
def drawDescriptiveStats(bucket: pd.Series, name: str, stats: dict, size_x: int, size_y: int) -> None:
    logger.debug("Function: drawDescriptiveStats")
    plt.figure(figsize=(size_x, size_y))
    plt.title(name)
    plt.plot(bucket, label="Data")
    plt.plot(stats["roll_mean"], label="1s mean")
    plt.plot(stats["auto_corr"], label="Auto Corr", color="greenyellow")
    plt.hlines(stats["mean"], xmin=bucket.first_valid_index(), xmax=bucket.last_valid_index(), label="Mean", color="magenta")
    plt.hlines(stats["median"], xmin=bucket.first_valid_index(), xmax=bucket.last_valid_index(), label="Median", color="royalblue")
    plt.plot(stats["min_idx"], stats["min"], "o", color="aqua", label="Minimum")
    plt.plot(stats["max_idx"], stats["max"], "o", color="crimson", label="Maximum")
    plt.plot([], [], ' ', label=f"""Standard deviation: {round(stats["std"], 2)}""")
    plt.plot([], [], ' ', label=f"""Variance: {round(stats["var"], 2)}""")
    plt.legend()
    plt.show()

# %%
def correlation(data: pd.DataFrame) -> pd.DataFrame:
    """Funkcja służąca do wyliczenia korelacji pomiędzy wszystkimi pięcioma kanałami

    Args:
        data (pd.DataFrame): dane wejściowe, w przypadku naszego projektu jest to cały dataset

    Returns:
        pd.DataFrame: Wyliczona korelacja pomiędzy wszystkimi kanałami w formie tabeli, tutaj pd.DataFrame
    """
    logger.debug("Function: correlation")
    corr = data.corr(method="pearson")
    return corr


# %%
def correlationHeatmap(calculated_correlation: pd.DataFrame, title: str, font_size: int ):
    """Metoda pozwalająca narysować mapę cieplną korelacji pomiędzy kanałamia

    Args:
        calculated_correlation (pd.DataFrame): obliczona korelacja jako dane wejściowe potrzebne zbudowania wykresu, tutaj powinien być to wynik metody correlation()
        title (str): Tytuł wyświetlany na heatmapie
        font_size (int): Wielkość fonta tytułu heatmapy
    """
    logger.debug("Function: correlationHeatmap")
    fig, ax = plt.subplots(figsize=(12,12))         # Sample figsize in inches
    map = sns.heatmap(calculated_correlation, square=True, linewidths=0.2, annot=True, cbar_kws={"orientation": "horizontal"})
    map.set(xlabel="Channel", ylabel="Channel")
    map.set_title(title, fontsize=font_size)
    plt.show()

# %%
def removeConstComp(data: pd.Series, method: str = "mean", window: int = SPS) -> pd.Series:
    """Function to remove contant component from pandas data series

    Args:
        data (pd.Series): data to remove constant component
        method (str): method of removal (mean, roll, diff)
        window (int): window size used in rolling method and diff

    Returns:
        pd.Series: data 
    """
    logger.debug("Function: removeConstComp")
    ret = pd.Series()
    
    logger.debug(f"Method: {method}")
    match method:
        case "mean":
            mean = data.mean()
            ret = data.sub(mean)    
        case "roll":
            rolling = data.rolling(window).mean()
            mean = data[0: rolling.first_valid_index() - 1].mean()
            rolling = rolling.fillna(mean)
            ret = data.sub(rolling)
        case "diff":
            ret = data.diff(window)           
    return ret

# %%
def autocorrelation(data) -> pd.DataFrame:
    """Function calculates autocorrelation of given data

    Args:
        data (pd.DataFrame, pd.Series): data to calculate autocorrelation

    Returns:
        pd.DataFrame: autocorrelation series, multiple series if input data had multiple series
    """
    logger.debug("Function: autocorrelation") 
    match type(data):
        case pd.DataFrame:
            logger.debug("Provided data is pandas DataFrame")
            ret = pd.DataFrame(columns= data.columns)
            logger.debug(f"Data columns: {data.columns}")
            for column in data.columns:
                ret[column] = pd.Series(sm.tsa.acf(data[column], nlags=data[column].size))                       
        
        case pd.Series:
            logger.debug("Provided data is pandas Series")
            ret = pd.DataFrame()
            if data.index.get_level_values(0).unique().dtype == 'object':
                logger.debug("Multiple buckets available")      
                logger.debug(f"Data buckets: {data.index.get_level_values(0).unique()}")
                ret["total"] = pd.Series(sm.tsa.acf(data, nlags=data.size))
                
                for bucket in data.index.get_level_values(0).unique():
                    ret[bucket] = pd.Series(sm.tsa.acf(data[bucket], nlags=data[bucket].size))
                    
            else:
                logger.debug("Single bucket available")  
                ret["total"] = sm.tsa.acf(data, nlags=data.size)
                
        case _:
            ret = None
        
    return ret

# %%
def drawAutocorrelation(data: pd.DataFrame, name: str = "Autocorrelation", overlaid = False, lineWidth: float = 1.0) -> None:
    """drawAutocorrelation fuction draw plot of auto correlation of provided data

    Args:
        data (pd.DataFrame): data to plot
        name (str, optional): title for plot. Defaults to "Autocorrelation".
        overlaid (bool, optional): draw multiple plots on one image. Defaults to False.
        lineWidth (float, optional): witdh of plot lines. Defaults to 1.0.
    """
    logger.debug(f"Function: drawAutocorrelation")
    logger.debug(data.columns)
    
    min_value = -0.2    
    for i, column in enumerate(data.columns, start= 1):            
            if data[column].min() < min_value:
                min_value = data[column].min()
    
    plt.figure(figsize=(15, 5))
    if data.columns.size == 1:
        logger.debug(f"Only one column")
        plt.plot(data, linewidth=lineWidth)
        plt.title(f"{name}")
        plt.ylim(min_value, 1.0)
        
    else:        
        logger.debug(f"Overlaid: {overlaid}")        
        if overlaid:
            if "total" in data.columns:
                logger.debug(f"`total` is one of colums")
                plt.subplot(1, 2, 1)
                plt.plot(data["total"], linewidth=lineWidth)
                plt.title(f"{name} total")
                plt.ylim(min_value, 1.0)
                
                plt.subplot(1, 2, 2)
                plt.title(f"""{name} {data.columns.where(data.columns != "total").dropna().values}""")
                plt.ylim(min_value, 1.0)
                
                for column in data.columns.where(data.columns != "total").dropna().values:
                    plt.plot(data[column], label=column, linewidth=lineWidth)
                
            else:
                logger.debug(f"`total` is not one of colums")
                
                plt.title(f"""{name} {data.columns.values}""")
                plt.ylim(min_value, 1.0)
                
                for column in data.columns:
                    plt.plot(data[column], label=column, linewidth=lineWidth)
        else:
            for i, column in enumerate(data.columns, start= 1):
                logger.debug(f"Column {column}, index {i}")              
                plt.subplot(1, data.columns.size, i)
                plt.ylim(min_value, 1.0)  
                plt.title(f"{name} {column}")
                plt.plot(data[column], linewidth=lineWidth)
                
    plt.legend()    
    plt.show()

# %%
def findMinimumsByAutoCorr(data: pd.DataFrame, analyze_ch: str = "ch1", window: int = SPS, order: int = 200, order2: int = 11, debug_draw: bool = False) -> list:
    """findMinimumsByAutoCorr find minimums in data based on autocorrelation

    Args:
        data (pd.DataFrame): data to analyze
        analyze_ch (str, optional): column from data to anazlyze. Defaults to "ch1".
        window (int, optional): rolling window mean size. Defaults to SPS.
        order (int, optional): samples to analyze, stage 1. Defaults to 200.
        order2 (int, optional): samples to analyze, stage 2. Defaults to 11.
        debug_draw (bool, optional): draw debug plot with minimums. Defaults to False.

    Returns:
        list: list of found minimums
    """    
    logger.debug(f"Function: findMinimumsByAutoCorr")
    
    if not isinstance(data, pd.DataFrame):
        logger.error(f"Bad input parameter: data")
        return None
    
    if not analyze_ch in data.columns:
        logger.error(f"Bad input parameter: analyze_ch")
        return None
    
    acorr = autocorrelation(data[analyze_ch])
    abs_acorr = acorr.abs()
    avr_abs_acorr = abs_acorr.rolling(window=window).mean() 
    local_min =  argrelextrema(avr_abs_acorr.values, np.less, order=order)[0]
    logger.debug(f"Local min: {local_min!r}")
    
    local_min2 =  argrelextrema(avr_abs_acorr["total"][local_min].values, np.less, order=order2)[0]
    logger.debug(f"Local min2: {local_min2!r}")
    
    tmp = [local_min[x] for x in local_min2]
    logger.debug(f"Tmp: {tmp}")
    
    if debug_draw:
        plt.figure(figsize=(15,5))
        plt.plot(abs_acorr)
        plt.plot(avr_abs_acorr)    
        plt.plot(tmp, avr_abs_acorr["total"][tmp], "o", color="red")
        plt.plot(data[analyze_ch])
        plt.show()
        
    return tmp
    
# %%
def __calculatePeriods(data: pd.Series, index: int, ret: list) -> None:
    logger.debug(f"Function: __calculatePeriods")
    
    size = data.size
    time = round(size/SPS, 4)    
    ret[index] = (size, time)
         
# %%
    
def calculatePeriods(data: pd.Series, buckets: list) -> dict:
    """calculatePeriods Funkcja wylicza okresy przediałów

    Args:
        data (pd.Series): Jedna kolumna danych
        buckets (list): Lista z przedziałami

    Returns:
        dict: Słownik z wyliczonymi wartościami
    """    
    logger.debug(f"Function: calculatePeriods")
    
    if not isinstance(data, pd.Series):
        logger.warning(f"Invalid data type, allowed type is: pd.Series, provided: {type(data)}")
        return None
    
    if not isinstance(buckets, list):
        logger.warning(f"Invalid data type, allowed type is: list, provided: {type(buckets)}")
        return None
    
    tmp = [None] * len(buckets)
    threads = [None] * len(buckets)
    for i, bucket in enumerate(buckets):      
        thr = th.Thread(target=__calculatePeriods, args=(data[bucket], i, tmp))
        threads[i] = thr
        thr.start()
        
    for thread in threads:
        thread.join()
       
    ret = dict().fromkeys(buckets)
    
    for i, bucket in enumerate(buckets):
        ret[bucket] = tmp[i]
        
    logger.debug(f"""Calculated periods:\n{tb(ret.values(), headers=["diff_samples", "diff_time_s"], tablefmt="fancy_grid", showindex="always")}""")
    return ret

# %%
def peaksPlot(data: pd.DataFrame, peaks: list,  title: str, x_label: str, y_label: str, plot_width: int, plot_height: int):
    """Metoda pozwalająca narysować wykres wraz z zaznaczonymi maximami.

    Args:
        data (pd.DataFrame): dane wejściowe, w przypadku naszego projektu jest to cały dataset
        peaks (list): lista punktów
        title (str): Tytuł na wykresie
        x_label (str): Etykieta osi X
        y_label (str): Etykieta osi Y
        plot_width (int): Szerokość wykresu
        plot_height (int): Wysokość wykresu
    """
    logger.debug(f"Function: peaksPlot")
    plt.figure(figsize=(plot_width,plot_height))
    plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(data[peaks], "x") 
    plt.show()

# %%
def __reduceResolution(data, column: str, index: int, ret: list, drop_by: int = SPS):
    logger.debug(f"Function: __reduceResolution")
    
    ret[index] = pd.Series(data[column][::drop_by])
# %%
def reduceResolution(data, drop_by: int = SPS):
    """reduceResolution drop resolution by drop_by parameter

    Args:
        data (pd.DataFrame or pd.Series): Data to reduce
        drop_by (int, optional): Reduce resolution by drop_by, like 300 / 5 = 60. Defaults to SPS.

    Returns:
        pd.DataFrame or pd.Series: Reduced data
    """
    logger.debug(f"Function: reduceResolution")
    
    if not ( isinstance(data, pd.DataFrame) or isinstance(data, pd.Series) ):
        logger.warning(f"Invalid data type, allowed type is: pd.DataFrame or pd.Series, provided: {type(data)}")
        return None
    
    if not isinstance(drop_by, int):
        logger.warning(f"Invalid data type, allowed type is: INT, provided: {type(drop_by)}")
        return None
        
    if isinstance(data, pd.Series):
        ret = pd.Series(data[::drop_by], index=range(data.size))
        logger.debug(f"Ret: {ret!r}")
        logger.debug(f"Index: {ret.keys()!r}")
        return ret
    
    else:        
        threads = [None] * data.columns.size
        th_data = [None] * data.columns.size
        
        for i, column in enumerate(data.columns):
            thr = th.Thread(target=__reduceResolution, args=(data, column, i, th_data, drop_by))
            threads[i] = thr
            thr.start()
            
        for thr in threads:
            thr.join()
        
        tmp = pd.DataFrame(th_data)
        ret = tmp.transpose()
        ret.columns = data.columns
        ret.index = range(ret[column].size)               
        logger.debug(f"Ret: {ret!r}")
        
        return ret
    
# %% 
def derivative(input_data: pd.DataFrame) -> pd.DataFrame:
    """Jest to funkcja, która liczy pochodną dla danego zbioru danych. Zwraca DataFrame. Funkcja jest potrzebna do dalszej analizy.

    Args:
        input_data (pd.DataFrame): Dataframe z danymi wejściowymi. Można też wprowadzić wybrany przedział
        

    Returns:
        pd.DataFrame: DataFrame z obliczoną pochodną
    """
    difference = input_data.diff()
    return difference

# %%
def __sampleWindow(data, column: str, index: int, ret: list, window: int = SPS):
    logger.debug(f"Function: __sampleWindow, thread: {index}")
    ret[index] = [data[column][i*window: (i+1)*window - 1].mean() for i in range(int(data[column].size / window))]
    
# %%
def sampleWindow(data, window: int = SPS):
    """sampleWindow, funkcja próbkuje dane co wartość podaną w window i liczy wartość średnią z utworzonych próbek

    Args:
        data (_type_): dane do przetworzenia
        window (int, optional): rozmiar okna do próbkowania. Defaults to SPS.

    Returns:
        pd.DataFrame lub pd.Series: DataFrame lub Seria zależnie od ilości kolumn w danych wejściowych
    """
    logger.debug(f"Function: sampleWindow")
    
    if not ( isinstance(data, pd.DataFrame) or isinstance(data, pd.Series) ):
        logger.warning(f"Invalid data type, allowed type is: pd.DataFrame or pd.Series, provided: {type(data)}")
        return None
    
    if not isinstance(window, int) and window > 1:
        logger.warning(f"Invalid data type, allowed type is: int, provided: {type(window)}")
        return None
    
    if isinstance(data, pd.Series):
        tmp = []
        for i in range(int(data.size / window)):
            tmp.append(data[i*window: (i+1)*window - 1].mean())
        logger.debug(i)
        ret = pd.Series(tmp)
        logger.debug(ret)
        
        return ret

    else:
        threads = [None] * data.columns.size
        th_data = [None] * data.columns.size
        
        for i, column in enumerate(data.columns):
            thr = th.Thread(target=__sampleWindow, args=(data, column, i, th_data, window))
            threads[i] = thr
            thr.start()
            
        for thr in threads:
            thr.join()
            
        tmp = pd.DataFrame(th_data)    
        ret = tmp.transpose()
        ret.columns = data.columns
        logger.debug(f"{ret!r}")
        
        return ret
            
# %%
def drawPlotXD(*args, over_laid: bool = True, width: int = 15, height: int = 5, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """drawPlotXD https://morganmlgman.duckdns.org/s/nhQEwxFKs#

    Args:
        over_laid (bool, optional): _description_. Defaults to True.
        width (int, optional): _description_. Defaults to 15.
        height (int, optional): _description_. Defaults to 5.
        xlabel (str, optional): _description_. Defaults to "".
        ylabel (str, optional): _description_. Defaults to "".
        title (str, optional): _description_. Defaults to "".
    """
    logger.debug(f"Function: drawPlotXD")
    
    if over_laid:
        plt.figure(figsize=(width, height))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        for item in args:
            if isinstance(item, dict):
                logger.debug(f"{item!r}")
                plt.plot(item["data"],
                         "o" if "draw_line" in item and not item["draw_line"] else "",
                         label=item["label"] if "label" in item else "",
                         color=item["color"] if "color" in item else "",
                         linewidth=item["line_width"] if "line_width" in item else 0.9,
                         marker=item["marker"] if "marker" in item else "")
            else:
                plt.plot(item, linewidth=0.9)
                
        plt.legend(loc="upper right")
        plt.show()
    
    else:
        plt.figure(figsize=(width, len(args)*height))
              
        for i, item in enumerate(args, start=1):
            plt.subplot(len(args), 1, i)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            if isinstance(item, dict):
                logger.debug(f"{item!r}")
                plt.plot(item["data"],
                         "o" if "draw_line" in item and not item["draw_line"] else "",
                         label=item["label"] if "label" in item else "",
                         color=item["color"] if "color" in item else "",
                         linewidth=item["line_width"] if "line_width" in item else 0.9,
                         marker=item["marker"] if "marker" in item else "")
            else:
                plt.plot(item, linewidth=0.9)
                
            plt.legend(loc="upper right")        
        plt.show()        

def cuttingZeroCount(input_data: pd.DataFrame):
    """Funkcja licząca liczbę przejść pochodnej przez 0. Można tu wrzucić pochodną

    Args:
        data (pd.DataFrame): Dane dla których zostanie obliczona liczba przejść przez 0. 
        Jeśli chcemy dać kolumnę, to trzeba ją wpisać, np. data["ch5"]
    Returns:
        counter (int): Liczba przejść przez zero
        points (list): punkty, w których doszło do tego przejścia
    """
    max = len(input_data)
    counter = 0
    points = []
    for i in range(1, (max-1)):
        if (input_data[i-1] < 0 and input_data[i+1] > 0) or (input_data[i-1] > 0 and input_data[i+1] < 0):
            counter = counter + 1
            points.append(i)

    return (counter, points)

# %%
def findOffsetByAutoCorr(data: pd.DataFrame, ch1: str, ch2: str, window: int = SPS, order: int = 300, order2: int = 11, debug_draw: bool = False, align: int = SPS) -> list:
    logger.debug(f"Function: findOffsetByAutoCorr")
    if not isinstance(data, pd.DataFrame):
        logger.warning(f"Incorrect data, allowed only pd.DataFrame, data: {data!r}, data_type: {type(data)}")
        raise TypeError(f"Incorrect data, allowed only pd.DataFrame, data_type: {type(data)}")
    
    data1_acorr_min = findMinimumsByAutoCorr(data, ch1, window, order, order2, debug_draw)
    data2_acorr_min = findMinimumsByAutoCorr(data, ch2, window, order, order2, debug_draw)
    
    data2_aligned = [None] * len(data1_acorr_min)
    
    for i, val1 in enumerate(data1_acorr_min):
        for j, val2 in enumerate(data2_acorr_min):
            if val1 < (val2 + align) and val1 > (val2 - align):
                data2_aligned[i] = val2
                
    logger.debug(data1_acorr_min)
    logger.debug(data2_aligned)

# %%    
def main(args = None):
    """Use logging insted of print for cleaner output
    """
    # --------------------------
    start_time = perf_counter()   
    logger.debug("Program beginning")
    # --------------------------
    
    data = pd.DataFrame(loadmat(DATA_FILE)[ARRAY_NAME], columns=(["ch1", "ch2", "ch3", "ch4", "ch5"]))
         
    findOffsetByAutoCorr(data, "ch4", "ch5", SPS, 300, 11, True, 2*SPS)
    
    logger.info(f"Run time {round(perf_counter() - start_time, 4)}s")
    
if __name__ == "__main__":
    main()
# %%
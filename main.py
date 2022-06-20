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
    plt.xlabel("Próbka")
    plt.ylabel("Wartość")
    plt.plot(bucket, label="Data", linewidth=0.7)
    plt.plot(stats["roll_mean"], label="1s mean")
    plt.hlines(stats["mean"], xmin=bucket.first_valid_index(), xmax=bucket.last_valid_index(), label="Mean", color="magenta")
    plt.hlines(stats["median"], xmin=bucket.first_valid_index(), xmax=bucket.last_valid_index(), label="Median", color="lime")
    plt.plot(stats["min_idx"], stats["min"], "o", color="aqua", label="Minimum")
    plt.plot(stats["max_idx"], stats["max"], "o", color="crimson", label="Maximum")
    plt.plot([], [], ' ', label=f"""Standard deviation: {round(stats["std"], 2)}""")
    plt.plot([], [], ' ', label=f"""Variance: {round(stats["var"], 2)}""")
    plt.legend(loc="upper right") 
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
                plt.plot(item, linewidth=0.7)
                
            plt.legend(loc="upper right")        
        plt.show()        

# %%

def cuttingZeroCount(input_data: pd.DataFrame):
    """Funkcja licząca liczbę przejść pochodnej przez 0. Można tu wrzucić pochodną

    Args:
        data (pd.DataFrame): Dane dla których zostanie obliczona liczba przejść przez 0. 
        Jeśli chcemy dać kolumnę, to trzeba ją wpisać, np. data["ch5"]
    Returns:
        counter (int): Liczba przejść przez zero
        points (list): punkty, w których doszło do tego przejścia
    """
    max = input_data.last_valid_index()
    counter = 0
    points = []
    for i in range(input_data.first_valid_index() + 1, (max-1)):
        if (input_data[i-1] < 0 and input_data[i+1] > 0) or (input_data[i-1] > 0 and input_data[i+1] < 0):
            counter = counter + 1
            points.append(i)

    return (counter, points)

# %%
def cuttingMeanCount(input_data: pd.DataFrame):
    """Funkcja licząca liczbę przejść pochodnej przez 0. Można tu wrzucić pochodną

    Args:
        data (pd.DataFrame): Dane dla których zostanie obliczona liczba przejść przez 0. 
        Jeśli chcemy dać kolumnę, to trzeba ją wpisać, np. data["ch5"]
    Returns:
        counter (int): Liczba przejść przez zero
        points (list): punkty, w których doszło do tego przejścia
    """
    max = input_data.last_valid_index()
    mean = input_data.mean()
    counter = 0
    points = []
    for i in range(input_data.first_valid_index() + 1, (max-1)):
        if (input_data[i-1] < mean and input_data[i+1] > mean) or (input_data[i-1] > mean and input_data[i+1] < mean):
            counter = counter + 1
            points.append(i)

    return (counter, points)
# %%
def findOffsetByAutoCorr(data: pd.DataFrame, ch1: str, ch2: str, window: int = SPS, order: int = 300, order2: int = 11, debug_draw: bool = False, align: int = SPS) -> float:
    """findOffsetByAutoCorr funkcja sprawdza przesunięcia czasowe pomiędzy kanałami danych 

    Args:
        data (pd.DataFrame): dane do sprawdzenia
        ch1 (str): kanał do porównania
        ch2 (str): kanał do porównania
        window (int, optional): okno do policzenia średniej. Defaults to SPS.
        order (int, optional): przedział do szukania minimum. Defaults to 300.
        order2 (int, optional): przedział2 do szukania minumum. Defaults to 11.
        debug_draw (bool, optional): czy rysować wykres pomocniczy. Defaults to False.
        align (int, optional): dopusczalne okno poszukiwań dopasowania. Defaults to SPS.

    Raises:
        TypeError: zły typ danych

    Returns:
        float: średnie przesunięcie danych w sekundach
    """
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
    offset = []        
    for i, item in enumerate(data2_aligned):
        if item:
            offset.append(data1_acorr_min[i] - item)
            
    logger.debug(offset)    
    sum = 0
    for item in offset:
        sum += item        
    mean = sum/len(offset)/SPS
    
    logger.debug(mean)
    
    return mean
# %%       
def timeIntervals(data: pd.DataFrame, column: str, ord: int = SPS) -> list:
    """Funkcja licząca czas trwania danej okresu. Przyjęto, że sekund ato 8192

    Args:
        data (pd.DataFrame): Plik wejściowy z danymi
        column (str): Kolumna w danych
        ord (int, optional): Do porównania n próbek ze sobą i szukania minimum. Im więcej, to dłużej trwa. Defaults to SPS.

    Returns:
        list: Lista z czasem trwania każdego okresu
    """
    SECOND = SPS

    tmp = []
    tmp = findMinimumsByAutoCorr(data, column, window=3*SPS, order = ord, order2=10, debug_draw=True)

    samples_number = len(tmp)

    values = []

    for i in range(1, samples_number):
        diff = tmp[i] - tmp[i-1]
        values.append(diff)

    time_diff = []
    for i in range(0,len(values)):
        val = values[i]/SECOND
       
        time_diff.append("{:.5f}".format(val))

    return time_diff   

# %%    
def main(args = None):
    """Use logging insted of print for cleaner output
    """
    # --------------------------
    start_time = perf_counter()   
    logger.debug("Program beginning")
    # --------------------------
    
    data = pd.DataFrame(loadmat(DATA_FILE)[ARRAY_NAME], columns=(["ch1", "ch2", "ch3", "ch4", "ch5"]))
    
    # INFO: Raw data
    
    # drawPlotXD(data["ch1"], xlabel="Próbka", ylabel="Wartość", title="Kanał CH1")
    # drawPlotXD(data["ch2"], xlabel="Próbka", ylabel="Wartość", title="Kanał CH2")
    # drawPlotXD(data["ch3"], xlabel="Próbka", ylabel="Wartość", title="Kanał CH3")
    # drawPlotXD(data["ch4"], xlabel="Próbka", ylabel="Wartość", title="Kanał CH4")
    # drawPlotXD(data["ch5"], xlabel="Próbka", ylabel="Wartość", title="Kanał CH5")
    
    # INFO: Punkt 2 - statystyki opisowe
    
    # ch1_stats = descriptiveStats(data["ch1"])
    # ch2_stats = descriptiveStats(data["ch2"])
    # ch3_stats = descriptiveStats(data["ch3"])
    # ch4_stats = descriptiveStats(data["ch4"])
    # ch5_stats = descriptiveStats(data["ch5"])
    
    # drawDescriptiveStats(data["ch1"], "Kanał CH1", ch1_stats, 15, 5)
    # print(f"""Kanał: CH1
    #       Średnia: {ch1_stats["mean"]}
    #       Mediana: {ch1_stats["median"]}
    #       Minimum: {ch1_stats["min"]}
    #       Maksimum: {ch1_stats["max"]}
    #       Odchylenie: {ch1_stats["std"]}
    #       Wariancja: {ch1_stats["var"]}
    #       """)
    
    # drawDescriptiveStats(data["ch2"], "Kanał CH2", ch2_stats, 15, 5)
    # print(f"""Kanał: CH2
    #       Średnia: {ch2_stats["mean"]}
    #       Mediana: {ch2_stats["median"]}
    #       Minimum: {ch2_stats["min"]}
    #       Maksimum: {ch2_stats["max"]}
    #       Odchylenie: {ch2_stats["std"]}
    #       Wariancja: {ch2_stats["var"]}
    #       """)
    
    # drawDescriptiveStats(data["ch3"], "Kanał CH3", ch3_stats, 15, 5)
    # print(f"""Kanał: CH3
    #       Średnia: {ch3_stats["mean"]}
    #       Mediana: {ch3_stats["median"]}
    #       Minimum: {ch3_stats["min"]}
    #       Maksimum: {ch3_stats["max"]}
    #       Odchylenie: {ch3_stats["std"]}
    #       Wariancja: {ch3_stats["var"]}
    #       """)
    
    # drawDescriptiveStats(data["ch4"], "Kanał CH4", ch4_stats, 15, 5)
    # print(f"""Kanał: CH4
    #       Średnia: {ch4_stats["mean"]}
    #       Mediana: {ch4_stats["median"]}
    #       Minimum: {ch4_stats["min"]}
    #       Maksimum: {ch4_stats["max"]}
    #       Odchylenie: {ch4_stats["std"]}
    #       Wariancja: {ch4_stats["var"]}
    #       """)
    
    # drawDescriptiveStats(data["ch5"], "Kanał CH5", ch5_stats, 15, 5)
    # print(f"""Kanał: CH5
    #       Średnia: {ch5_stats["mean"]}
    #       Mediana: {ch5_stats["median"]}
    #       Minimum: {ch5_stats["min"]}
    #       Maksimum: {ch5_stats["max"]}
    #       Odchylenie: {ch5_stats["std"]}
    #       Wariancja: {ch5_stats["var"]}
    #       """)
    
    # INFO: Punkt 3 - korelacja 
    
    # data_corr = correlation(data)
    # correlationHeatmap(data_corr, "Korelacja", 16)
    
    # off_ch1_ch2 = findOffsetByAutoCorr(data, "ch1", "ch2", 3*SPS, SPS, 10, SPS // 3)
    # off_ch1_ch3 = findOffsetByAutoCorr(data, "ch1", "ch3", 3*SPS, SPS, 10, SPS // 3)
    # off_ch1_ch4 = findOffsetByAutoCorr(data, "ch1", "ch4", 3*SPS, SPS, 10, SPS // 3)
    # off_ch1_ch5 = findOffsetByAutoCorr(data, "ch1", "ch5", 3*SPS, SPS, 10, SPS // 3)
    
    # off_ch2_ch3 = findOffsetByAutoCorr(data, "ch2", "ch3", 3*SPS, SPS, 10, SPS // 3)
    # off_ch2_ch4 = findOffsetByAutoCorr(data, "ch2", "ch4", 3*SPS, SPS, 10, SPS // 3)
    # off_ch2_ch5 = findOffsetByAutoCorr(data, "ch2", "ch5", 3*SPS, SPS, 10, SPS // 3)
    
    # off_ch3_ch4 = findOffsetByAutoCorr(data, "ch3", "ch4", 3*SPS, SPS, 10, SPS // 3)
    # off_ch3_ch5 = findOffsetByAutoCorr(data, "ch3", "ch5", 3*SPS, SPS, 10, SPS // 3)
    
    # off_ch4_ch5 = findOffsetByAutoCorr(data, "ch4", "ch5", 3*SPS, SPS, 10, SPS // 3)
    
    # print(f"""
    #       Przesunięcie ch1 <-> ch2: {off_ch1_ch2} sekund
    #       przesunięcie ch1 <-> ch3: {off_ch1_ch3} sekund
    #       przesunięcie ch1 <-> ch4: {off_ch1_ch4} sekund
    #       przesunięcie ch1 <-> ch5: {off_ch1_ch5} sekund
          
    #       przesunięcie ch2 <-> ch3: {off_ch2_ch3} sekund
    #       przesunięcie ch2 <-> ch4: {off_ch2_ch4} sekund
    #       przesunięcie ch2 <-> ch5: {off_ch2_ch5} sekund
          
    #       przesunięcie ch3 <-> ch4: {off_ch3_ch4} sekund
    #       przesunięcie ch3 <-> ch5: {off_ch3_ch5} sekund
          
    #       przesunięcie ch4 <-> ch5: {off_ch4_ch5} sekund
    #       """)
    
    # INFO: Punkt 4 - okresowość
    
    # drawAutocorrelation(autocorrelation(data), "Autokorelacja", True, 0.8)
    
    # min_ch1 = findMinimumsByAutoCorr(data, "ch1", 3*SPS, SPS, 10, True)
    # min_ch2 = findMinimumsByAutoCorr(data, "ch2", 3*SPS, SPS, 10, True)
    # min_ch3 = findMinimumsByAutoCorr(data, "ch3", 3*SPS, SPS, 10, True)
    # min_ch4 = findMinimumsByAutoCorr(data, "ch4", 3*SPS, SPS, 10, True)
    # min_ch5 = findMinimumsByAutoCorr(data, "ch5", 3*SPS, SPS, 10, True)
    
    # ch1_split, ch1_keys = dataSplit(data, min_ch1, "ch1")
    # ch2_split, ch2_keys = dataSplit(data, min_ch2, "ch2")
    # ch3_split, ch3_keys = dataSplit(data, min_ch3, "ch3")
    # ch4_split, ch4_keys = dataSplit(data, min_ch4, "ch4")
    # ch5_split, ch5_keys = dataSplit(data, min_ch5, "ch5")
    
    # print(calculatePeriods(ch1_split, ch1_keys))
    # print(calculatePeriods(ch2_split, ch2_keys))
    # print(calculatePeriods(ch3_split, ch3_keys))
    # print(calculatePeriods(ch4_split, ch4_keys))
    # print(calculatePeriods(ch5_split, ch5_keys))
    
    # INFO: Punkt 5 
    
    ch1_sample_mean = []
    ch1_sample_var = []
    ch1_zero_cross = []
    ch1_mean_cross = []
    
    for i in range(0, data["ch1"].size, SPS):
        sample = data["ch1"][i: i+SPS]
        ch1_sample_mean.append(sample.mean())
        ch1_sample_var.append(sample.var())
        
        (val, point) = cuttingZeroCount(sample.diff())
        ch1_zero_cross.append(val)
        
        (val2, point2) = cuttingMeanCount(sample)
        ch1_mean_cross.append(val2)
        
    drawPlotXD(ch1_sample_mean, xlabel="Próbka", ylabel="Wartość", title="Średnia z próbki CH1")
    drawPlotXD(ch1_sample_var, xlabel="Próbka", ylabel="Wartość", title="Wariancja z próbki CH1")
    drawPlotXD(ch1_zero_cross, xlabel="Próbka", ylabel="Wartość", title="Przejście przez 0 pochodnej z próbki CH1")
    drawPlotXD(ch1_mean_cross, xlabel="Próbka", ylabel="Wartość", title="Przejście przez średnią wartości z próbki CH1")
    
    
    logger.info(f"Run time {round(perf_counter() - start_time, 4)}s")
    
if __name__ == "__main__":
    main()
# %%


import FileReader as f
import pandas as pd
import Regression as r
from SQL import PersistenceUtil
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg
import time
import Visualisation as visual

import numpy as np

class FilterVisualizer(visual.Visualizer):
    '''
    A visualizer object for the research task.
    
    '''
    def additional_elements(self, index, min_y):
        pass

def convolution(y, length, mode="same"):
    '''Calculate convolution between a function and an impuls of specified 
    length.
    
    Args:
        y (array_like): The function to convolve with.
        length (int): Length of the impulse. Must be greater 0.
    
    Returns
        array_like: The convolution result.

    '''
    y2 = np.ones(length)/length
    y = np.convolve(y, y2, mode=mode)
    
    return y

def regression(y):
    '''Calculates kernel regression for a function of continuous data.
    
    Args:
        y (array_like): The function for the kernel regression.
    
    Returns
        ndarray: The regression result for the mean.

    '''
    kr = KernelReg(y,y.index,'c')
    y, y_std = kr.fit(y.index)

    return y

def hodrick_prescott(y):
    '''Separates a function into trend and cyclic component.
    
    It uses lambda = 1,600 for the HP filter.
    
    Args:
        y (array_like): The function to be split.
    
    Returns
        ndarray: The estimated trend in the function provided.
        ndarray: The estimated cycle in the function provided.

    '''
    cycle, trend = sm.tsa.filters.hpfilter(y, 1600)

    return cycle, trend

def wrangling(df_train, df_ideal, title="Training Data", plot=False):
    '''Applies the data matching operations from the training data task to a 
    given data frame.
    
    Args:
       df_train: The dataframe that acts as training data.
       df_ideal: The ideal functions to get mapped to.
       title (str): Title of the chart and name of the output file. Defaults 
           to `Training Data`.
       plot (bool): Whether a plot should be generated. Defaults to False.
       
    
    Returns
        pd.DataFrame: The result data for this matching process.

    '''
    df_train.index.names = ['x']
    df_ideal.index.names = ['x']
    regression = r.Regression(df_ideal)
    df_train_mse = df_train.apply(regression.mse)
    df_result = regression.min_value(df_train_mse, df_train)
    chosen_ideal = df_ideal[df_result.min_y.tolist()]

    if plot:
        visual.PlotFactory.show(title, df_train, df_result, chosen_ideal)

    return df_result

def main():
    '''
    This is the main function for the research task.
    
    '''
    # Get the data from files.
    pu = PersistenceUtil()
    df_train = pu.read_df("train").set_index("X");
    df_ideal = pu.read_df("ideal").set_index("X");

    
    ####################
    # Convolution
    ####################
    df_train_convolve = df_train.copy()
    support_len = 12
    start = time.time()

    df_train_convolve = df_train_convolve.apply(lambda y: \
                                                convolution(y, support_len))
    print("Convolution", time.time() - start)

    i = int(support_len/2)
    df_train_convolve = df_train_convolve.iloc[i:-i]
    df_result = wrangling(df_train, df_ideal)
    df_ideal_convolve = df_ideal.iloc[i:-i]
    df_result_convolve = wrangling(df_train_convolve, df_ideal_convolve, \
                                   title="Convolution", plot=True)


    ####################
    # Regression
    ####################
    df_train_regression = df_train.copy()
    start = time.time()
    df_train_regression = df_train_regression.apply(regression)
    print("Regression", time.time() - start)
    df_result_regression = wrangling(df_train_regression, df_ideal, \
                                     title="Regression", plot=True)



    ####################
    # Hodrick-Prescott
    ####################
    df_train_hp = df_train.copy()
    start = time.time()
    df_train_hp_trend = df_train_hp.apply(lambda y: hodrick_prescott(y)[1])
    print("Hodrick-Prescott", time.time() - start)

    # This is the result with original ideal functions.
    df_result_hp = wrangling(df_train_hp_trend, df_ideal, \
                             title="Hodrick-Prescott", plot=True)

    # Do the same now by applying the HP filter to the ideal functions as well.
    start = time.time()
    df_ideal_hp_trend = df_ideal.apply(lambda y: hodrick_prescott(y)[1])
    df_result_hp_trend = wrangling(df_train_hp_trend, df_ideal_hp_trend, \
                                   title="Hodrick-Prescott (Ideal Trend)", \
                                   plot=True)
    print("Hodrick-Prescott (Trend)", time.time() - start)

    

    ####################
    # Display results
    ####################
    print("\nOriginal\n", df_result)
    print("\nConvolution\n", df_result_convolve)
    print("\nKernel Regression\n", df_result_regression)
    print("\nHodrick-Prescott\n", df_result_hp)
    print("\nHodrick-Prescott (Trend)\n", df_result_hp_trend)

    print("Done")

if __name__ == '__main__':
    main()
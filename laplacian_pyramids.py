############################# IMPORTING ######################################
import numpy as np
import pandas as pd
import sys
import scipy.io
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import normalize
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from numpy.linalg import norm 

##############################################################################

def rolling_window(a, window):
    """
    Create a rolling window view of an input numpy array.

    Parameters:
    a (numpy.ndarray): Input array for which rolling windows are to be created.
    window (int): Size of the rolling window.

    Returns:
    numpy.ndarray: Rolling window view of the input array `a`.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def create_frame(series,window):
    """
    Convert a series into a DataFrame of rolling windows.

    Parameters:
    series (pandas.Series): Input series to convert into rolling windows.
    window (int): Size of the rolling window.

    Returns:
    pandas.DataFrame: DataFrame where each column represents a rolling window of the input series.
    """
    amplitude = np.array(series.values.flatten().tolist())
    Location_amplitude = rolling_window(amplitude, window)
    intervals_lst = []
    for j in range(len(Location_amplitude)):
        extrected_values = Location_amplitude[j].flatten().tolist()
        intervals_lst.append(extrected_values)
    Location_amplitude = pd.concat([pd.Series(x) for x in intervals_lst], axis=1).transpose()
    return Location_amplitude

def get_df_and_epsilons(data,window,split):
    """
    Split the data into training and testing DataFrames and calculate epsilon.

    Parameters:
    data (pandas.DataFrame): DataFrame containing the data to split.
    window (int): Size of the rolling window.
    split (int): Start index for splitting into training and testing.

    Returns:
    pandas.DataFrame, pandas.DataFrame, float: Training DataFrame, Testing DataFrame, and Epsilon value.
    """
    amplitude = np.array(data.values.flatten().tolist())
    sigma = calcEpsilon(range(len(amplitude)), amplitude)
    df = create_frame(data,window)
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train, test, sigma

def calcEpsilon(X, dataList):
    """
    Calculate the epsilon value for a Gaussian kernel based on the given dataset.

    Parameters:
    X (list or range): List of indices or range representing positions in the dataset.
    dataList (numpy.ndarray): Dataset for which epsilon needs to be calculated.

    Returns:
    float: Epsilon value calculated based on the dataset and epsilon_factor.
    """
    
    epsilon_factor = 10
    
    dist = []
    for x in X:
        dist.append([])
        for y in X:
            _x = np.array(dataList[x])
            _y = np.array(dataList[y])
            dist[x].append(LA.norm(_x - _y))
    temp = list(dist + np.multiply(np.identity(len(X)), max(max(dist))))
    mins = []
    for row in temp:
        small = sys.maxsize
        for el in row:
            if(el < small and el != 0):
                small = el
        mins.append(small)
    return max(mins) * epsilon_factor

def kernel_matrix(dataframe, sigma, is_zero):
    """
    Calculate a row-normalized kernel matrix using the Gaussian (RBF) kernel.

    The kernel matrix is computed based on pairwise Euclidean distances between data points.
    Optionally, the diagonal of the kernel matrix can be set to zero.

    Parameters:
    dataframe (pd.DataFrame): A DataFrame where each row represents a data point and each column represents a feature.
    sigma (float): The standard deviation parameter for the Gaussian (RBF) kernel. Determines the width of the kernel.
    is_zero (bool): If True, sets the diagonal elements of the kernel matrix to zero. If False, retains the diagonal elements.

    Returns:
    numpy.ndarray: A row-normalized kernel matrix where each element represents the similarity between data points.
    """
    if is_zero:
        dataframe = dataframe.iloc[:, :-1].to_numpy()  # Convert DataFrame to a NumPy array, excluding the last column
        pairwise_dists = pdist(dataframe, 'sqeuclidean')  # Compute pairwise squared Euclidean distances
        pairwise_dists = squareform(pairwise_dists)  # Convert distance vector to a full distance matrix
        kernel = np.exp(-pairwise_dists / sigma**2)  # Apply Gaussian (RBF) kernel function
        np.fill_diagonal(kernel, 0)  # Set diagonal elements of the kernel matrix to zero
        normalize_kernel = normalize(kernel, axis=1, norm='l1')  # Normalize rows of the kernel matrix
        return normalize_kernel
    else:
        dataframe = dataframe.iloc[:, :-1].to_numpy()  # Convert DataFrame to a NumPy array, excluding the last column
        pairwise_dists = pdist(dataframe, 'sqeuclidean')  # Compute pairwise squared Euclidean distances
        pairwise_dists = squareform(pairwise_dists)  # Convert distance vector to a full distance matrix
        kernel = np.exp(-pairwise_dists / sigma**2)  # Apply Gaussian (RBF) kernel function
        normalize_kernel = normalize(kernel, axis=1, norm='l1')  # Normalize rows of the kernel matrix
        return normalize_kernel

def LP_Train(dataframe, f, sigma_0, is_zero):
    """
    Train a Laplacian Pyramids (LP) model by computing a multi-scale approximation of the data using Gaussian (RBF) kernels.

    This function performs iterative training to build a series of approximations of the target values
    using different scales (sigma values) of the Gaussian kernel. It returns the multi-scale approximations,
    the differences between approximations, and other relevant information for evaluation.

    Parameters:
    dataframe (pd.DataFrame): A DataFrame where each row represents a data point, and the last column contains the target labels.
    sigma_0 (float): The initial standard deviation for the Gaussian (RBF) kernel.
    is_zero (bool): If True, sets the diagonal elements of the kernel matrix to zero. If False, retains the diagonal elements.

    Returns:
    tuple: A tuple containing:
        - f_multiscale (list of numpy.ndarray): A list of multi-scale approximations of the target labels.
        - f_approx (list of numpy.ndarray): A list of approximations at each scale.
        - d (list of numpy.ndarray): A list of differences between the target labels and the approximations.
        - err (list of float): A list of norms of the differences at each scale.
        - lst_sigma (list of float): A list of sigma values used in each iteration.
        - f (numpy.ndarray): The target labels.
    """
    f_multiscale = []  # List to store multi-scale approximations
    f_approx = []      # List to store approximations at each scale
    d = []             # List to store differences between target labels and approximations
    lst_sigma = []     # List to store sigma values used in each iteration
    counter = 0
    sigma = sigma_0
    
    while counter <= 20:
        if counter == 0:
            # First iteration: compute kernel matrix (K0), f0, and d1
            k0 = kernel_matrix(dataframe, sigma, is_zero)  # Compute kernel matrix
            f0 = k0.dot(f)  # Compute approximation f0
            di = f - f0  # Compute difference d1
            f_multiscale.append(f0)
            f_approx.append(f0)
            d.append(di)
            lst_sigma.append(sigma_0)
        else:
            sigma = sigma / 2  # Reduce sigma
            lst_sigma.append(sigma)
            ki = kernel_matrix(dataframe, sigma, is_zero)  # Compute kernel matrix at new scale
            approx_i = ki.dot(di)  # Compute new approximation
            f_approx.append(approx_i)
            fi = [sum(i) for i in zip(*f_approx)]  # Aggregate approximations
            f_multiscale.append(fi)
            di = f - fi  # Update difference
            d.append(di)
        
        counter += 1
    
    err = [norm(i) for i in d]
    
    return f_multiscale, f_approx, d, err, lst_sigma

def LP(dataframe_train, f, dataframe_test, sigma_0, is_zero):
    """
    Perform prediction using the LP model trained on the training dataset.

    This function uses the trained LP model to predict values for a test dataset based on the best scale
    found during training. It computes the predicted values by evaluating each test sample against the
    multi-scale approximations obtained from the training phase.

    Parameters:
    dataframe_train (pd.DataFrame): A DataFrame where each row represents a data point, and the last column contains the target labels for training.
    dataframe_test (pd.DataFrame): A DataFrame where each row represents a data point for which predictions are to be made.
    sigma_0 (float): The initial standard deviation for the Gaussian (RBF) kernel.
    is_zero (bool): If True, sets the diagonal elements of the kernel matrix to zero. If False, retains the diagonal elements.

    Returns:
    list: A list of predicted values for each sample in the test dataset.
    """
    f_multiscale, f_approx, d, err, lst_sigma = LP_Train(dataframe_train, f, sigma_0, is_zero)  # Train the model
    index_min = min(range(len(err)), key=err.__getitem__)  # Find the index with the minimum error
    predicted_values = []
    
    for sample in dataframe_test.iloc[:, :-1].values:
        sample = np.array(sample.tolist()).reshape(1, -1)  # Reshape test sample
        pairwise_dists = cdist(sample, dataframe_train, 'sqeuclidean').reshape(-1)  # Compute pairwise distances
        f_new = []
        
        for i in range(0, index_min + 1):
            kernel_vector = np.exp(-pairwise_dists / (lst_sigma[i]**2))  # Compute kernel vector
            normalize_vector = kernel_vector / kernel_vector.sum(axis=0, keepdims=1)  # Normalize kernel vector
            
            if np.isnan([normalize_vector]).sum() > 0:
                normalize_vector = kernel_vector
            else:
                normalize_vector = normalize_vector
            
            if i == 0:
                f0_new = normalize_vector.dot(f)  # Compute approximation for the first scale
                f_new.append(f0_new)
            else:
                di_new = normalize_vector.dot(d[i])  # Compute difference for subsequent scales
                f_new.append(di_new)
        
        predicted_values.append(sum(f_new))  # Aggregate predictions
    
    return predicted_values


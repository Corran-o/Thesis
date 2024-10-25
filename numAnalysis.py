import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quaternion as qt
import scipy.signal as sc

def accDataNumpy(df):
    accelData = df[['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy()
    return accelData

def gyrDataNumpy(df):
    gyrData = df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy()
    return gyrData

def quatDataNumpy(df):
    quatData = df[['Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']].to_numpy()
    return quatData

def basicStats(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std

def peakIdxs(data):
    peaksIdx, _ = sc.find_peaks(data, distance=10)
    return peaksIdx

def crossCorr(data1, data2): # Normalised cross correlation, Input: 2 1D arrays
    corr = np.correlate(data1 - np.mean(data1), data2 - np.mean(data2))
    return corr

def rangeOfMotion(data):
    rom = np.ptp(data, axis=0)
    return rom
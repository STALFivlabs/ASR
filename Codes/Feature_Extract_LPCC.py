#calculate LPCC coefficients from LPC coefficients

import numpy as np
import math

def normi(arr):
    maxi = np.max(arr) 
    mini = np.min(arr)
    diffr = maxi - mini
    for i in range(len(arr[:, 0])):
        for k in range(len(arr[0, :])):
            arr[i][k] = (arr[i][k] - mini) / diffr
    return arr

def lpcc(s, fs, seq, order):

    #divide into segments of 25 ms with overlap of 10ms
    nSamples = np.int32(0.025*fs)
    overlap = np.int32(0.01*fs)
    nFrames = np.int32(np.ceil(len(s)/(nSamples-overlap)))

    #zero padding to make signal length long enough to have nFrames
    padding = ((nSamples-overlap)*nFrames) - len(s)
    if padding > 0:
        signal = np.append(s, np.zeros(padding))
    else:
        signal = s
    segment = np.empty((nSamples, nFrames))
    start = 0
    for i in range(nFrames):
        segment[:,i] = signal[start:start+nSamples]
        start = (nSamples-overlap)*i

    lpcc_coeffs = np.empty((order, nFrames))
    for n in range(1, order + 1):
        # Use order + 1 as upper bound for the last iteration
        upbound = (order + 1 if n > order else n)
        lpcc_coef = -sum(i * seq[:, n - i - 1] for i in range(1, upbound)) * 1. / upbound
        lpcc_coef -= seq[:, n - 1] if n <= len(seq[:, n]) else 0
        np.vstack((lpcc_coeffs[:, n], lpcc_coef))
    lpcc_coeffs = np.nan_to_num(lpcc_coeffs)
    lpcc_coeffs = np.float64(lpcc_coeffs)
    lpcc_coeffs = normi(lpcc_coeffs)
    
    return lpcc_coeffs.T

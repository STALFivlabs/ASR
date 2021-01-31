#calculate LPC coefficients from sound file

import numpy as np
import math



def auto_correlation(x, lag):
  n = len(x)
  x_mean = np.mean(x)
  y = x - x_mean
  r = np.zeros((lag+1,1))
  for k in range(lag+1):
    top = 0
    bottom = 0
    for i in range(n-k):
      top += y[i]*y[i+k]
    for i in range(n):
      bottom += y[i]*y[i]
    if bottom==0:
      r[k][0] = 0.0001
    else:  
      r[k][0] = top/bottom
  return r

"""Function to calculate symmetric matrix for matrix multiplication in further step to calculate alpha."""

def sym_matrix(n):
  size = (n,n)
  ret_mat = np.zeros(size)
  for i in range(0,n):
    for j in range(0,n):
      ret_mat[i][j] = (np.abs(i-j))
  
  return ret_mat

"""#Function to calculate Linear Prediction Coefficient."""

def lpc(signal, sampling_freq, no_coeff):
  #print('Stage1: Time Framing')
  
  #Time Framing: Standard-25ms
  no_sample = int(0.025*sampling_freq)
  no_delay = int(0.010*sampling_freq)
  no_frame = int(math.ceil(len(signal)/(no_sample-no_delay)))

  #Padding
  padding = ((no_sample - no_delay)*no_frame) - len(signal)
  if padding > 0:
    s = np.append(signal, np.zeros(padding))
  else:
    s = signal
  #print('Stage1: Done')
  
  
  #print('Stage2: Segmentation')
  #segmenting signal in frames
  start = 0
  count = 0
  for i in range(no_frame):
    if start == 0:
      seg_mat = np.zeros((1, no_sample))
      seg_mat[0] = s[start:no_sample]
      start = no_sample
      count += 1
    else:
      if s.shape[0] - start >= 1200:
        temp_mat = np.zeros((1, no_sample))
        temp_mat[0][:] = s[start-no_delay:start-no_delay+no_sample]
        start = start - no_delay + no_sample
        seg_mat = np.vstack((seg_mat, temp_mat))
        count += 1
  #print('Stage2: Done')
  #Hamming Window operation (Optional)
  x = np.zeros_like(seg_mat)
  for i in range(count):
    x = seg_mat[i]*np.hamming(no_sample)

  #print('Stage3: LPC with Yule-walker algorithm')
  #print('Please Wait')
  #calculating LPC with Yule-walker algorithm
  lpc_coeff = np.zeros((count, no_coeff))
  for i in range(count):
    r1 = auto_correlation(seg_mat[i], no_coeff)
    temp = r1[1:][0]
    r = np.resize(temp,(no_coeff,1))
    r = (-1)*r
    R = sym_matrix(no_coeff)
    for j in range(no_coeff):
      for k in range(no_coeff):
        val = int(R[k][j])
        R[k][j] = r1[val]
    pro_mat = np.zeros((no_coeff,1))
    pro_mat = np.dot(np.linalg.pinv(R),r)
    lpc_coeff[i] = np.resize(pro_mat, (1, no_coeff)) 
    #Converting in the range -1 to +1
    lpc_coeff[i] = lpc_coeff[i][:]/np.max(np.abs(lpc_coeff[i]))

  #print('Stage3: Done')
  return lpc_coeff

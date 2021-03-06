from Feature_Extract_LPC import lpc
from Feature_Extract_LPCC import lpcc
from Feature_Match_LBG import lbg_codebook
from Feature_Match_LBG import EuclideanDistance
from training import train

import numpy as np
from scipy.io.wavfile import read        
        
def closestUser(features, codebooks):
    minDistance = np.inf
    for name in codebooks.keys():
        dist = EuclideanDistance(features, codebooks[name])
        Distance = np.sum(np.min(dist, axis = 1))/(np.shape(dist)[0])
        if Distance < minDistance:
            minDistance = Distance
            speaker = name

    return speaker



no_filtbank = 12
orderLPC = 15

names = ['Siddharth', 'Thanmay', 'Luqman', 'Fauzan']
#names.append('Apoorva')


codebooks = {}

#TRAINING
print("TRAINING\n")
for name in names:
    codebooks[name] = train(no_filtbank, orderLPC, name, 2*len(names), 'lpc')

#COMPARISON
print("COMPARISON\n")
directory = 'test'


no_crct=0
no_crct_c=0
for name in names:
    fname = '\\' + name + '_1.wav' 
    print ('Now ', name+'\'s test features are being tested')
    (fs,s) = read(directory + fname)
    lpc_coeff = lpc(s, fs, orderLPC)
    lpcc_coeff = lpcc(s, fs, lpc_coeff.T, orderLPC)
    foundname = closestUser(lpc_coeff, codebooks)
    foundname_c = closestUser(lpcc_coeff, codebooks)
    print (name+'_1.wav', ' in test matches with speaker', foundname+' in train (LPC) \n')
    if(name == foundname):
        no_crct+=1
    print (name+'_1.wav', ' in test matches with speaker', foundname_c+' in train (LPCC) \n')
    if(name == foundname_c):
        no_crct_c+=1

#RESULT
print("RESULT\n")
accuracy = (no_crct/len(codebooks.keys()))*100
print("Accuracy (LPC): ",accuracy, "%")
accuracy_c = (no_crct_c/len(codebooks.keys()))*100
print("Accuracy (LPCC): ",accuracy_c, "%")

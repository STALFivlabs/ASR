from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import EUDistance
from mel_coefficients import mfcc
from LPC import lpc
from train import training

nSpeaker = 8
nfilterbank = 12
orderLPC = 15
(codebooks_mfcc, codebooks_lpc) = training(nfilterbank, orderLPC)
directory = 'C:\\Users\\Luqman\\Desktop\\ASR\\train'
fname = str()
nCorrect_MFCC = 0
nCorrect_LPC = 0
codeface = dict()

def MiniDist(features, codebooks):
    speaker = 0
    mindist = np.inf
    for k in range(np.shape(codebooks)[0]):
        D = EUDistance(features, codebooks[k,:,:])
        dist = np.sum(np.min(D, axis = 1))/(np.shape(D)[0])
        if dist < mindist:
            mindist = dist
            speaker = k
    return speaker

for i in range(nSpeaker):
    fname = '/s' + str(i+1) + '.wav'
    print ('Now speaker ', str(i+1), 'features are being tested')
    (fs,s) = read(directory + fname)
    mel_coefs = mfcc(s,fs,nfilterbank)
    lpc_coefs = lpc(s, fs, orderLPC)
    sp_mfcc = MiniDist(mel_coefs, codebooks_mfcc)
    sp_lpc = MiniDist(lpc_coefs, codebooks_lpc)

    print ('Speaker', (i+1), ' in test matches with speaker ', (sp_mfcc+1), 'in train for training with MFCC')
    print ('Speaker', (i+1), ' in test matches with speaker ', (sp_lpc+1), 'in train for training with LPC')

    if i == sp_mfcc:
        nCorrect_MFCC += 1
    if i == sp_lpc:
        nCorrect_LPC += 1


percentCorrect_MFCC = (nCorrect_MFCC/nSpeaker)*100
print ('Accuracy of result for training with MFCC is ', percentCorrect_MFCC, '%')
percentCorrect_LPC = (nCorrect_LPC/nSpeaker)*100
print ('Accuracy of result for training with LPC is ', percentCorrect_LPC, '%')

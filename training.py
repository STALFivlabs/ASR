#training function for ASR. Assumed method is MFCC. Assumed 2 functions exist i.e mfcc() and lbg()
import numpy as np
from scipy.io.wavfile import read
# !I couldnt find if imports are required for pathlib and sndhdr. You can check.
def training(no_fltr_bnks, directory): # !directory to be passed as str(some folder name containing the sound files)
  #code to find number of speakers by looping through the directory !make this as a general function in start of this ASR program and then use it, including general formula of centroids' number
  no_spkrs = 0
  for path in pathlib.Path("directory").iterdir(): #this goes thru all the files in the directory
    if path.is_file() and sndhdr.what(path)(0) = 'wav': #checks whether that file in the directory is actually a valid wav file
        no_spkrs += 1
  no_centroids = no_spkrs*2
  #actual training:
  ctr=0 #to keep track of successful finish !kinda redundant but still
  codebooks = np.empty((no_spkrs, no_fltr_bnks, no_centroids)) #the main codebooks vector
  print("Starting training for" + no_spkrs + "speakers")
  for path in pathlib.Path("directory").iterdir():
    if path.is_file() and sndhdr.what(path)(0) = 'wav': # !sndhdr.what() returns a 5 tuple, first value of which is the audio type. So I indexed (0). Please check.
      sample_rate, signal = read(path) #for each file, read() returns a tuple. The first one is samples/second (sample rate) and second is the actual data read from the audio file !(so I just called it 'signal' but not sure if its correct)
      codebooks[i] = lbg(mfcc(signal, sample_rate, no_fltr_bnks), no_centroids) #features passed to lbg are the MFCCs for each speaker. !assumed appropriate functions are called mfcc and lbg 
      ctr++
  if(ctr==no_spkrs): # !could also use len(codebooks) or something so we dont neeed ctr, but codebooks was defined already with a set length?
    print("Training finished!")
  else:
    print("Uh-oh. Training was interrupted. Please check directory.")

  return codebooks;

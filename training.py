#training function for ASR. Assumed method is MFCC. Assumed 2 functions exist i.e mfcc() and lbg()
import numpy as np
from scipy.io.wavfile import read
# !I couldnt find if imports are required for pathlib and sndhdr. You can check.
def training(directory, model_type, no_fltr_bnks = 0, order_LPC = 0): # model_type accepts 'mfcc' or 'lpc' as valid arguments and trains according to that model only !directory to be passed as str(some folder name containing the sound files)
  #code to find number of speakers by looping through the directory !make this as a general function in start of this ASR program and then use it, including general formula of centroids' number
  no_spkrs = 0
  for path in pathlib.Path("directory").iterdir(): #this goes thru all the files in the directory
    if path.is_file() and sndhdr.what(path)(0) == 'wav': #checks whether that file in the directory is actually a valid wav file
        no_spkrs += 1
  no_centroids = no_spkrs*2
  #actual training:
  ctr=0 #to keep track of successful finish !kinda redundant but still
  codebooks = np.empty((no_spkrs, no_centroids, no_fltr_bnks)) #the main codebooks vector
  print("Starting training for" + no_spkrs + "speakers")
  for path in pathlib.Path("directory").iterdir():
    if path.is_file() and sndhdr.what(path)(0) == 'wav': # !sndhdr.what() returns a 5 tuple, first value of which is the audio type. So I indexed (0). Please check.
      sample_rate, signal = read(path) #for each file, read() returns a tuple. The first one is samples/second (sample rate) and second is the actual data read from the audio file !(so I just called it 'signal' but not sure if its correct)
      if model_type == 'mfcc':
        if ctr==0:
          codebooks = np.empty((no_spkrs, no_centroids, no_fltr_bnks)) #the main codebooks vector
        codebooks[i] = lbg(mfcc(signal, sample_rate, no_fltr_bnks), no_centroids) #features passed to lbg are the MFCCs for each speaker. !assumed appropriate functions are called mfcc and lbg 
      elif model_type == 'lpc':
        if ctr==0:
          codebooks = np.empty((no_spkrs, order_LPC, no_centroids)) #the main codebooks vector
        codebooks[i] = lbg(mfcc(signal, sample_rate, order_LPC), no_centroids)
      else:
        print("Invalid model type! Model type should be 'mfcc' or 'lpc'.")
        sys.exit()
      ctr++
  if(ctr==no_spkrs): # !could also use len(codebooks) or something so we dont neeed ctr, but codebooks was defined already with a set length?
    print("Training finished!")
  else:
    print("Uh-oh. Training was interrupted. Please check directory.")
    sys.exit()
  
  #writing the result to a file:
  f=open('Training_result.txt','w')
  for elem in codebooks:
    f.write(elem+'\n')
  f.close()

  return codebooks

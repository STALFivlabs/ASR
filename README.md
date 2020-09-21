# Automatic Speaker Recognition

**Description**

This project aims to apply some basic techniques in signal processing for speaker recognition.

The following is the project workflow:
1. Pre-processing of input audio signal
2. Feature Extraction with LPC
3. Feature Matching with LBG
4. Dataset Training
5. User Matching

Do feel free to let us know your feedback through the comments section. Thanks!

**Compatibility**

This code has been developed and tested well in Windows in the Python IDLE environment.

### Instructions

- Download our GitHub repository from [here](https://github.com/STALFivlabs/ASR). Run the _ASR.py_ file as is.
- To add your own sample, record two seperate voice recordings directly in .wav file format (dont convert it from one type to .wav). Store these two recordings in the test and train folders with the naming convention. Then modify the _names_ list in the _ASR.py_ file accordingly.

>**Special Notes**
> - To fix the warning of a singular matrix whose determinant is zero, we have modified it to peform calculations on pseudo inverse (pinv). Additionally, we have set _r[k][0] = 0.0001_ due to NaN issues coupled with the previous warning.
> Using the above patch may or may not result in loss of accuracy.
> - Conversion from LPC to LPCC has been performed but results are not as expected. Although, feel free to modify the code for testing purposes. Let us know your results !

### Results

On a database size of 4 speakers, the LPC and LPCC algorithms have observed 100% and upto 50% accuracy respectively. However, it is expected to decrease with more number of samples.

**References**
- LPC [literature](http://www.practicalcryptography.com/miscellaneous/machine-learning/tutorial-cepstrum-and-lpccs/)
- Auto Correlation technique was taken [from here](https://www.philippe-fournier-viger.com/spmf/TimeSeriesAutocorellation.php)
- K-means clustering [reference](https://www.youtube.com/watch?v=1XqG0kaJVHY&feature=youtu.be)
- This is our reference [code flow](https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf)

**Demonstration**

Here's a simple demonstration of the code at run time.

> ![](asrdemo.gif)

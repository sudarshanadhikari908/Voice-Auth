import os


import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
path ="Sound/"
#we shall store all the file names in this list
filelist = []

for root, dirs, files in os.walk(path):
	for file in files:
        #append the file name to the list
		filelist.append(os.path.join(root,file))

#print all the file names
for name in filelist:
    print(name)

    ipd.Audio(name)
    # load audio files with librosa
    signal, sr = librosa.load(name)


    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
    mfccs.shape

    plt.figure(figsize=(25, 10))
    librosa.display.specshow(mfccs,
                             x_axis="time",
                             sr=sr)
    plt.colorbar(format="%+2.f")
    plt.show()
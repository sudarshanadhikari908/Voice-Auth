import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr
import librosa.display
import IPython.display as ipd
import os


path = ["./Sounds/bed"]
# ,"./Sounds/cat"
# ,"./Sounds/cat"
# path = np.array(path)


def read(a):
    files = fnmatch.filter(os.listdir(a), "*.wav")

    print(len(files))
    return files
i = 0
while i < len(path):
    # print(path[i])
    a = path[i]
    b = read(a)
    i += 1
b = np.array(b)
b = np.reshape(b, (2, 1007))


print(type(b))
print (b)
for y in b:


    # ipd.Audio(y, rate=22050)
    scale, sr = lr.load(y)
    filter_banks = lr.filters.mel(n_fft=2048, sr=22050, n_mels=10)
    filter_banks.shape
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(filter_banks,
                             sr=sr,
                             x_axis="linear")
    plt.colorbar(format="%+2.f")
    plt.show()
    mel_spectrogram = lr.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)

    mel_spectrogram.shape

    log_mel_spectrogram = lr.power_to_db(mel_spectrogram)
    log_mel_spectrogram.shape

    plt.figure(figsize=(25, 10))
    librosa.display.specshow(log_mel_spectrogram,
                             x_axis="time",
                             y_axis="mel",
                             sr=sr)
    plt.colorbar(format="%+2.f")
    plt.show()






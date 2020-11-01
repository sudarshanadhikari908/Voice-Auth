import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr
import librosa.display
import IPython.display as ipd
import os


path = ["./Sounds/bed","./Sounds/cat"]
# path = np.array(path)

# print (len(path))


# path = os.path.append(path1, path2)

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

print(type(b))
print (b)
for y in b:


    ipd.Audio(y, rate=22050)
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






# print(files)


# for root, dirs, files in os.walk(".", topdown=False):
#    for name in files:
#       a= (os.path.join(root, name))
#       a = np.array(a, dtype=float)
#
#    for name in dirs:
#       b= (os.path.join(root, name))
#       b = np.array(b, dtype=float)
# files = [file for file in files if file[-1:] == '.wav']
# print(files[0])
#
# for y in files:
#
#
#     ipd.Audio(files, rate=22050)
#     scale, sr = lr.load(files)
#     filter_banks = lr.filters.mel(n_fft=2048, sr=22050, n_mels=10)
#     filter_banks.shape
#     plt.figure(figsize=(25, 10))
#     librosa.display.specshow(filter_banks,
#                              sr=sr,
#                              x_axis="linear")
#     plt.colorbar(format="%+2.f")
#     plt.show()
#     mel_spectrogram = lr.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
#
#     mel_spectrogram.shape
#
#     log_mel_spectrogram = lr.power_to_db(mel_spectrogram)
#     log_mel_spectrogram.shape
#
#     plt.figure(figsize=(25, 10))
#     librosa.display.specshow(log_mel_spectrogram,
#                              x_axis="time",
#                              y_axis="mel",
#                              sr=sr)
#     plt.colorbar(format="%+2.f")
#     plt.show()
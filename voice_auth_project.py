import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr

data_dir = './Sounds/bed/'
audio_files = glob(data_dir + '/*.wav')
a = len(audio_files)
print(a)
audio, sfreq = lr.load(audio_files[0])
time =np.arange(0, len(audio))/ sfreq
print(time)

fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time(s)', ylabel='Sound Amplitude')
plt.show()

for files in range(0, len(audio_files), 1):
    audio, sfreq = lr.load(audio_files[0])
    time = np.arange(0, len(audio)) / sfreq
    print(time)

    fig, ax = plt.subplots()
    ax.plot(time, audio)
    ax.set(xlabel='Time(s)', ylabel='Sound Amplitude')
    plt.show()






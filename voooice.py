import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr
import librosa.display
import IPython.display as ipd


data_dir = "Sounds/bed/"
audio_files = glob(data_dir + '/*.wav')

a = len(audio_files)
print(a)
audio, sfreq = lr.load(audio_files[0])
time = np.arange(0, len(audio)) / sfreq
print(time)

fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time(s)', ylabel='Sound Amplitude')
plt.show()
for files in range(0, len(audio_files), 1):


    ipd.Audio(data_dir)
    scale, sr = lr.load(data_dir)
    filter_banks = lr.filters.mel(n_fft=2048, sr=22050, n_mels=10)
    filter_banks.shape
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(filter_banks,
                             sr=sr,
                             x_axis="linear")
    plt.colorbar(format="%+2.f")
    plt.show()
    mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)

    mel_spectrogram.shape

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    log_mel_spectrogram.shape

    plt.figure(figsize=(25, 10))
    librosa.display.specshow(log_mel_spectrogram,
                             x_axis="time",
                             y_axis="mel",
                             sr=sr)
    plt.colorbar(format="%+2.f")
    plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr
import librosa.display
import IPython.display as ipd

pathAudio = 'Sounds'
files = librosa.util.find_files(pathAudio, ext=['WAV'])
files = np.array(files, dtype=float)
sr = 22050

for y in files:


    ipd.Audio(files, rate=22050)
    scale, sr = lr.load(files)
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
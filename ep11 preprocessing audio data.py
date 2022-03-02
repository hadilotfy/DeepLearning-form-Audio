import librosa,librosa.display
import matplotlib.pyplot as plt
import numpy as np
#import scipy as sp
from sklearn.metrics import homogeneity_completeness_v_measure

file = "quran.wav"

# Waveform
signal, sr = librosa.load(file, sr=22050)  # sr * T -> ss050 * 4
# librosa.display.waveshow(signal,sr=sr) 
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()

#fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0,sr,len(magnitude))
left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(magnitude)/2)]

# plt.plot(left_frequency,left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

#stft -> specetogram
hop_length = 512
n_fft = 2048
stft = librosa.core.stft(signal,n_fft=n_fft,hop_length=hop_length)
spectogram = np.abs(stft)
log_spectogram = librosa.amplitude_to_db(spectogram)
# librosa.display.specshow(log_spectogram,sr=sr,hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

#MFCCs
MFCCs = librosa.feature.mfcc(signal,n_fft=n_fft,hop_length=hop_length,n_mfcc =45)
librosa.display.specshow(MFCCs,sr=sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCCs")
plt.colorbar()
plt.show()


import numpy as np
from scipy.io import wavfile

soundfile='./dataset/짧은_총장전소리.wav'
sample_rate, signal = wavfile.read(soundfile)
signal = np.array([x.mean() for x in signal])
new_name=soundfile.split('/')
new_name[-1]='mono'+new_name[-1]
new_name='/'.join(new_name)
wavfile.write(new_name,sample_rate,signal.astype(np.int16))
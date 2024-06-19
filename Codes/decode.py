import zlib
import numpy as np
from scipy.io import wavfile
from scipy.fft import ifft
import matplotlib.pyplot as plt

name='sin_mono_shorten'
soundfile=f'./result/decoded_{name}.wav'
originalfile=f'./dataset/{name}.wav'
fft_compress_file=f'./compressed/fft_compress_{name}.bin'
fft_compress_file_error=f'./compressed/fft_compress_file_error_{name}.bin'
result_csv=f'result_{name}.csv'
def mean_squared_error(original, reconstructed):
    error = np.mean((original - reconstructed)**2)
    return error
def mean_absolute_error(original, reconstructed):
    error = np.mean(np.abs(original - reconstructed))
    return error

def decoding_fft(soundfile,originalfile,fft_compress_file,fft_compress_file_error,result_csv):
    with open(fft_compress_file,'rb') as f:
        byte_data=f.read()
        fft_data = str(zlib.decompress(byte_data))[2:-1].split()
        digit_fft= int(fft_data[0])
        sample_rate= int(fft_data[1])
        fft_data=np.array([complex(float(x.split(',')[0])*10**(-digit_fft),float(x.split(',')[1])*10**(-digit_fft)) for x in fft_data[2:]])
        ifft_data=ifft(fft_data)

    with open(fft_compress_file_error,'rb') as f:
        byte_data=f.read()
        data_temp=str(zlib.decompress(byte_data))[2:-1].split()
        digit_error=int(data_temp[0])
        fft_data_error=np.array([float(x)*10**(-digit_error) for x in data_temp[1:]])
    reconstructed_data=ifft_data.real+fft_data_error
    sample_rate_origin, original_signal = wavfile.read(originalfile)
    original_signal=np.array([x.mean() for x in original_signal])
    mse=mean_squared_error(original_signal,reconstructed_data)
    mae=mean_absolute_error(original_signal,reconstructed_data)
    print(f'MSE Error: {mse}')
    print(f'MAE Error: {mae}')
    with open(result_csv,'at') as f:
        f.write(f',{mse},{mae}\n')

    # t = np.linspace(0., len(reconstructed_data) / sample_rate, len(reconstructed_data))
    # plt.figure(figsize=(12, 6))
    # plt.plot(t, original_signal, label='original_data')
    # plt.plot(t,reconstructed_data,label='reconstructed_data',linestyle='--')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title('Original Signal vs. Reconstructed Signal')
    # plt.legend()
    # plt.show()

    wavfile.write(soundfile,sample_rate,reconstructed_data.astype(np.int16))

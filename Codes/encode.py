import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile  # scipy.io.wavfile에서 read 함수 import
from scipy.fft import fft,ifft
import zlib
import os


def mean_squared_error(original, reconstructed):
    error = np.mean((original - reconstructed)**2)
    return error
def mean_absolute_error(original, reconstructed):
    error = np.mean(np.abs(original - reconstructed))
    return error
def get_wavsize(soundfile):
    sample_rate, original_signal = wavfile.read(soundfile)
    return len(original_signal)

# soundfile='./dataset/제일짧은칼빼는소리.wav'
# fft_compress_file='./compressed/fft_compress.bin'
# fft_compress_file_error='./compressed/fft_compress_file_error.bin'

def encoding_fft(soundfile, fft_compress_file, fft_compress_file_error, result_csv, N):
    sample_rate, original_signal = wavfile.read(soundfile)  # scipy.io.wavfile 모듈에서 read 함수 사용
    #original_signal = np.array([x.mean() for x in original_signal])

    #압축 인자 
    N=min(N,len(original_signal)-1) #최소 진폭에서 몇 번째를 한계 진폭수
    digit_fft=0 #fft 결과 진폭을 소숫점 몇번째까지 자를 것인지
    digit_error=0

    fft_data=fft(original_signal)
    Amps=sorted(fft_data,key=lambda x:abs(x))

    Limit_Amp=Amps[N]
    fft_data=[complex(round(x.real,digit_fft),round(x.imag,digit_fft)) if abs(x)>=abs(Limit_Amp) else complex(0) for x in fft_data]
    # temp=fft_data[:]

    # fft_data=fft_data[:len(fft_data)//2+1]
    # fft_data=fft_data+fft_data[-2::-1]
    # with open('test.txt','wt') as f:
    #     for i in range(len(fft_data)):
    #         f.write(str(temp[i])+str(fft_data[i])+'\n')
    ifft_data=ifft(fft_data)

    with open(fft_compress_file,'wb') as f:
        fft_save_data= f'{digit_fft} '+f'{sample_rate} '+ ' '.join([f'{x.real//10**(-digit_fft)},{x.imag//10**(-digit_fft)}' for x in fft_data])
        f.write(zlib.compress(fft_save_data.encode(encoding='utf-8'),level=9))
    with open(fft_compress_file_error,'wb') as f:
        fft_save_data_error=f'{digit_error} '+ ' '.join(str(int(round(x,digit_error)//10**(-digit_error))) for x in (original_signal-ifft_data.real))
        f.write(zlib.compress(fft_save_data_error.encode(encoding='utf-8'),level=9))

    print(f'File Name:{soundfile}')
    print(f'FFT MSE Error:{mean_squared_error(original_signal,ifft_data.real)}')
    print(f'FFT MAE Error:{mean_absolute_error(original_signal,ifft_data.real)}')

    soundfile_size=os.path.getsize(soundfile)
    fft_compress_file_size=os.path.getsize(fft_compress_file)
    fft_compress_file_error_size=os.path.getsize(fft_compress_file_error)


    print(f'\nOriginal Size:\t{soundfile_size}')
    print(f'FFT Size:\t{fft_compress_file_size}\n\t\t{fft_compress_file_error_size}')
    compressed_rate=(fft_compress_file_size+fft_compress_file_error_size)/(soundfile_size)
    print(f'FFT Compressed Rate:{compressed_rate}')
    with open(result_csv, 'at') as f:
        f.write(f'{N},{compressed_rate}')
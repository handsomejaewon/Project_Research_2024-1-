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


soundfile='./dataset/제일짧은칼빼는소리.wav'
fft_compress_file='./compressed/fft_compress.bin'
fft_compress_file_error='./compressed/fft_compress_file_error.bin'
lpc_compress_file='./compressed/lpc_compress.bin'
lpc_compress_file_error='./compressed/lpc_compress_file_error.bin'

sample_rate, original_signal = wavfile.read(soundfile)  # scipy.io.wavfile 모듈에서 read 함수 사용
#original_signal = np.array([x.mean() for x in original_signal])

#압축 인자 
N=min(0,len(original_signal)-1) #최소 진폭에서 몇 번째를 한계 진폭수
print(f'Audio Len:{len(original_signal)}')
digit_fft=0 #fft 결과 진폭을 소숫점 몇번째까지 자를 것인지
digit_error=0
p = 10 # 예측 계수 개수

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



t = np.linspace(0., len(original_signal) / sample_rate, len(original_signal))

# LPC 예측 계수 계산 함수 (최소자승법 사용)
def calculate_lpc_coefficients(signal, p):
    n = len(signal)
    A = np.zeros((n - p, p))
    for i in range(p, n):
        A[i - p] = signal[i - p:i][::-1]
    b = signal[p:]
    coefficients = np.linalg.lstsq(A, b, rcond=None)[0]
    return coefficients

# 예측 신호 생성 함수
def predict_signal(coefficients, signal, p):
    n = len(signal)
    predicted_signal = np.zeros(n)
    predicted_signal[:p] = signal[:p]  # 초기값은 원본 신호 값 사용
    for i in range(p, n):
        predicted_signal[i] = np.dot(coefficients, signal[i - p:i][::-1])
    return predicted_signal

# LPC 계수 계산

lpc_coefficients = calculate_lpc_coefficients(original_signal, p)


# 예측 신호 생성
lpc_predicted_signal = predict_signal(lpc_coefficients, original_signal, p)

# 원본 신호와 예측 신호 비교
plt.figure(figsize=(12, 6))
plt.plot(t, original_signal, label='Original Signal')
plt.plot(t, lpc_predicted_signal, label='Predicted Signal', linestyle='--')
plt.plot(t, ifft_data.real, label='ifft_data', linestyle=':')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original Signal vs. Predicted Signal')
plt.legend()
plt.show()


with open(fft_compress_file,'wb') as f:
    fft_save_data= f'{digit_fft} '+f'{sample_rate} '+ ' '.join([f'{x.real//10**(-digit_fft)},{x.imag//10**(-digit_fft)}' for x in fft_data])
    f.write(zlib.compress(fft_save_data.encode(encoding='utf-8'),level=9))
with open(lpc_compress_file,'wb') as f:
    lpc_save_data= ' '.join(str(x) for x in lpc_coefficients)
    f.write(zlib.compress(lpc_save_data.encode(encoding='utf-8'),level=9))
with open(fft_compress_file_error,'wb') as f:
    fft_save_data_error=f'{digit_error} '+ ' '.join(str(int(round(x,digit_error)//10**(-digit_error))) for x in (original_signal-ifft_data.real))
    f.write(zlib.compress(fft_save_data_error.encode(encoding='utf-8'),level=9))
with open(lpc_compress_file_error,'wb') as f:
    lpc_save_data_error=' '.join(str(int(round(x,digit_error)//10**(-digit_error))) for x in (original_signal-lpc_predicted_signal))
    f.write(zlib.compress(lpc_save_data_error.encode(encoding='utf-8'),level=9))


print(f'File Name:{soundfile}')
print(f'LPC MSE Error:{mean_squared_error(original_signal,lpc_predicted_signal)}')
print(f'LPC MAE Error:{mean_absolute_error(original_signal,lpc_predicted_signal)}')
print(f'FFT MSE Error:{mean_squared_error(original_signal,ifft_data.real)}')
print(f'FFT MAE Error:{mean_absolute_error(original_signal,ifft_data.real)}')

soundfile_size=os.path.getsize(soundfile)
lpc_compress_file_size=os.path.getsize(lpc_compress_file)
lpc_compress_file_error_size=os.path.getsize(lpc_compress_file_error)
fft_compress_file_size=os.path.getsize(fft_compress_file)
fft_compress_file_error_size=os.path.getsize(fft_compress_file_error)


print(f'\nOriginal Size:\t{soundfile_size}')
print(f'LPC Size:\t{lpc_compress_file_size}\n\t\t{lpc_compress_file_error_size}')
print(f'FFT Size:\t{fft_compress_file_size}\n\t\t{fft_compress_file_error_size}')
print(f'LPC Compressed Rate:{(lpc_compress_file_size+lpc_compress_file_error_size)/(soundfile_size)}')
print(f'FFT Compressed Rate:{(fft_compress_file_size+fft_compress_file_error_size)/(soundfile_size)}')
from encode import encoding_fft,get_wavsize
from decode import decoding_fft
name='제일짧은칼빼는소리'
soundfile=f'./dataset/{name}.wav'
fft_compress_file=f'./compressed/fft_compress_{name}.bin'
fft_compress_file_error=f'./compressed/fft_compress_file_error_{name}.bin'
result_csv=f'./result/result_{name}.csv'

resultfile=f'./result/decoded_{name}.wav'
fft_compress_file=f'./compressed/fft_compress_{name}.bin'
fft_compress_file_error=f'./compressed/fft_compress_file_error_{name}.bin'

with open(result_csv,'wt') as f:
    f.write(f'FILE NAME,{name}.wav\n')
    f.write('N,compressed rate,MSE,MAE\n')
for N in range(0,get_wavsize(soundfile),1):
    print(N)
    encoding_fft(soundfile,fft_compress_file,fft_compress_file_error,result_csv,N)
    decoding_fft(resultfile,soundfile,fft_compress_file,fft_compress_file_error,result_csv)
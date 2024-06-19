import numpy as np
from scipy.io.wavfile import write

# 복잡한 파형 생성 함수
def generate_complex_tone(frequencies, amplitudes, duration, samplerate=44100):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    wave = np.zeros_like(t)
    for freq, amp in zip(frequencies, amplitudes):
        wave += amp * np.sin(2 * np.pi * freq * t)
    # 정규화 (최대 진폭이 1을 넘지 않도록)
    wave = wave / np.max(np.abs(wave))
    return wave

# 샘플 속성 설정
samplerate = 44100  # 샘플 레이트 (Hz)
duration = 2  # 지속 시간 (초)

# 가청 주파수 범위에서 1000개의 주파수 선택
np.random.seed(42)  # 결과 재현성을 위한 시드 설정
frequencies = np.random.uniform(20, 20000, 1000)

# 각각의 주파수에 대해 랜덤한 진폭 생성
amplitudes = np.random.uniform(1, 3, 1000)

# 복잡한 파형 생성
complex_tone = generate_complex_tone(frequencies, amplitudes, duration, samplerate)

# WAV 파일로 저장
output_wav_path = "complex_tone_mono.wav"
write(output_wav_path, samplerate, complex_tone.astype(np.float32))

output_wav_path
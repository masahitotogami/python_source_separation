
#wave形式の音声波形を読み込むためのモジュール(wave)をインポート
import wave as wave

#numpyをインポート（波形データを2byteの数値列に変換するために使用）
import numpy as np

#可視化のためにmatplotlibモジュールをインポート
import matplotlib.pyplot as plt

#sounddeviceモジュールをインポート
import sounddevice as sd

#読み込むサンプルファイル
sample_wave_file="./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"

#ファイルを読み込む
wav=wave.open(sample_wave_file)

#PCM形式の波形データを読み込み
data=wav.readframes(wav.getnframes())

#dataを2バイトの数値列に変換
data=np.frombuffer(data, dtype=np.int16)

#dataを再生する
sd.play(data,wav.getframerate())

print("再生開始")

#再生が終わるまで待つ
status = sd.wait()
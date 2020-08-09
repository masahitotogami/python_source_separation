
#wave形式の音声波形を読み込むためのモジュール(wave)をインポート
import wave as wave

#numpyをインポート（波形データを2byteの数値列に変換するために使用）
import numpy as np

#可視化のためにmatplotlibモジュールをインポート
import matplotlib.pyplot as plt

#sounddeviceモジュールをインポート
import sounddevice as sd

#録音する音声データの長さ (秒)
wave_length=5

#サンプリング周波数
sample_rate=16000

print("録音開始")

#録音開始
data = sd.rec(int(wave_length*sample_rate),sample_rate, channels=1)

#録音が終了するまで待つ
sd.wait() 

#2バイトのデータとして書き込むためにスケールを調整 
data_scale_adjust=data*np.iinfo(np.int16).max

#2バイトのデータに変換
data_scale_adjust=data_scale_adjust.astype(np.int16)

#waveファイルに書き込む
wave_out = wave.open("./record_wave.wav", 'w')

#モノラル:1、ステレオ:2
wave_out.setnchannels(1)

#サンプルサイズ2byte
wave_out.setsampwidth(2)

#サンプリング周波数
wave_out.setframerate(sample_rate)

#データを書き込み
wave_out.writeframes(data_scale_adjust)

#ファイルを閉じる
wave_out.close()

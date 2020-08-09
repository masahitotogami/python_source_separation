
#wave形式の音声波形を読み込むためのモジュール(wave)をインポート
import wave as wave

#numpyをインポート（波形データを2byteの数値列に変換するために使用）
import numpy as np

#scipyのsignalモジュールをインポート（stft等信号処理計算用)
import scipy.signal as sp

#読み込むサンプルファイル
sample_wave_file="./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"

#ファイルを読み込む
wav=wave.open(sample_wave_file)

#PCM形式の波形データを読み込み
data=wav.readframes(wav.getnframes())

#dataを2バイトの数値列に変換
data=np.frombuffer(data, dtype=np.int16)

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)

#短時間フーリエ変換後のデータ形式を確認
print("短時間フーリエ変換後のshape: ",np.shape(stft_data))

#周波数軸の情報
print("周波数軸 [Hz]: ",f)

#時間軸の情報
print("時間軸[sec]: ",t)

#waveファイルを閉じる
wav.close()

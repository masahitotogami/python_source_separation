

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

#時間領域の波形に戻す
t,data_post=sp.istft(stft_data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)

#2バイトのデータに変換
data_post=data_post.astype(np.int16)

#waveファイルに書き込む
wave_out = wave.open("./istft_post_wave.wav", 'w')

#モノラル:1、ステレオ:2
wave_out.setnchannels(1)

#サンプルサイズ2byte
wave_out.setsampwidth(2)

#サンプリング周波数
wave_out.setframerate(wav.getframerate())

#データを書き込み
wave_out.writeframes(data_post)

#ファイルを閉じる
wave_out.close()

#waveファイルを閉じる
wav.close()
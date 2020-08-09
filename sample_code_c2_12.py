

#wave形式の音声波形を読み込むためのモジュール(wave)をインポート
import wave as wave

#numpyをインポート（波形データを2byteの数値列に変換するために使用）
import numpy as np

#scipyのsignalモジュールをインポート（stft等信号処理計算用)
import scipy.signal as sp

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

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)

#特定の周波数成分を消す(50番目の周波数よりも低い周波数成分を全て消す)
stft_data[:50,:]=0


#時間領域の波形に戻す
t,data_post=sp.istft(stft_data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)

#2バイトのデータに変換
data_post=data_post.astype(np.int16)

#dataを再生する
sd.play(data_post,wav.getframerate())

print("再生開始")

#再生が終わるまで待つ
status = sd.wait()

#waveファイルを閉じる
wav.close()

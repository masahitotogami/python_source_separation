

#wave形式の音声波形を読み込むためのモジュール(wave)をインポート
import wave as wave

#numpyをインポート（波形データを2byteの数値列に変換するために使用）
import numpy as np

#scipyのsignalモジュールをインポート（stft等信号処理計算用)
import scipy.signal as sp

#sounddeviceモジュールをインポート
import sounddevice as sd

#乱数の種を設定
np.random.seed(0)

#読み込むサンプルファイル
sample_wave_file="./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"

#ファイルを読み込む
wav=wave.open(sample_wave_file)

#音声区間の長さを取る
n_speech=wav.getnframes()

#サンプリングレートを取る
sampling_rate=wav.getframerate()

#PCM形式の波形データを読み込み
speech_signal=wav.readframes(wav.getnframes())

#speech_signalを2バイトの数値列に変換
speech_signal=np.frombuffer(speech_signal, dtype=np.int16)


#音声データに白色雑音を混ぜる

#雑音だけの区間のサンプル数を設定
n_noise_only=40000

#全体の長さ
n_sample=n_noise_only+n_speech

#白色雑音を生成
wgn_signal=np.random.normal(scale=0.04,size=n_sample)

#2バイトのデータとして書き込むためにスケールを調整 
wgn_signal=wgn_signal*np.iinfo(np.int16).max

#2バイトのデータに変換
wgn_signal=wgn_signal.astype(np.int16)

#白色雑音を混ぜる
mix_signal=wgn_signal
mix_signal[n_noise_only:]+=speech_signal

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(mix_signal,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)

#入力信号の振幅を取得
amp=np.abs(stft_data)

#入力信号の位相を取得
phase=stft_data/np.maximum(amp,1.e-20)

#雑音だけの区間のフレーム数
n_noise_only_frame=np.sum(t<(n_noise_only/sampling_rate))

#スペクトルサブトラクションのパラメータ
p=1.0
alpha=2.0

#雑音の振幅を推定
noise_amp=np.power(np.mean(np.power(amp,p)[:,:n_noise_only_frame],axis=1,keepdims=True),1./p)


#入力信号の振幅の1%を下回らないようにする
eps=0.01*np.power(amp,p)

#出力信号の振幅を計算する
processed_amp=np.power(np.maximum(np.power(amp,p)-alpha*np.power(noise_amp,p),eps), 1./p)

#出力信号の振幅に入力信号の位相をかける
processed_stft_data=processed_amp*phase

#時間領域の波形に戻す
t,processed_data_post=sp.istft(processed_stft_data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)

#2バイトのデータに変換
processed_data_post=processed_data_post.astype(np.int16)

#waveファイルに書き込む
wave_out = wave.open("./process_wave_ss.wav", 'w')

#モノラル:1、ステレオ:2
wave_out.setnchannels(1)

#サンプルサイズ2byte
wave_out.setsampwidth(2)

#サンプリング周波数
wave_out.setframerate(wav.getframerate())

#データを書き込み
wave_out.writeframes(processed_data_post)

#ファイルを閉じる
wave_out.close()

import matplotlib.pyplot as plt 


#スペクトログラムをプロットする
fig=plt.figure(figsize=(10,4))

#スペクトログラムを表示する
spectrum,  freqs, t, im=plt.specgram(processed_data_post,NFFT=512,noverlap=512/16*15,Fs=wav.getframerate(),cmap="gray")

#カラーバーを表示する
fig.colorbar(im).set_label('Intensity [dB]')

#x軸のラベル
plt.xlabel("Time [sec]")

#y軸のラベル
plt.ylabel("Frequency [Hz]")

#音声ファイルを画像として保存
plt.savefig("./spectrogram_ss_result.png")

#画像を画面に表示
plt.show()


#雑音込みの入力信号も時間領域の波形に戻す
t,data_post=sp.istft(stft_data,fs=wav.getframerate(),window="hann",nperseg=512,noverlap=256)

#2バイトのデータに変換
data_post=data_post.astype(np.int16)

#スペクトログラムをプロットする
fig=plt.figure(figsize=(10,4))

#スペクトログラムを表示する
spectrum,  freqs, t, im=plt.specgram(data_post,NFFT=512,noverlap=512/16*15,Fs=wav.getframerate(),cmap="gray")

#カラーバーを表示する
fig.colorbar(im).set_label('Intensity [dB]')

#x軸のラベル
plt.xlabel("Time [sec]")

#y軸のラベル
plt.ylabel("Frequency [Hz]")

#音声ファイルを画像として保存
plt.savefig("./spectrogram_noisy.png")

#画像を画面に表示
plt.show()

#waveファイルに書き込む
wave_out = wave.open("./input_wave.wav", 'w')

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







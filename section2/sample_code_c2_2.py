#!/usr/bin/env python3

#wave形式の音声波形を読み込むためのモジュール(wave)をインポート
import wave as wave

#numpyをインポート（波形データを2byteの数値列に変換するために使用）
import numpy as np

#可視化のためにmatplotlibモジュールをインポート
import matplotlib.pyplot as plt

#読み込むサンプルファイル
sample_wave_file="./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"

#ファイルを読み込む
wav=wave.open(sample_wave_file)

#PCM形式の波形データを読み込み
data=wav.readframes(wav.getnframes())

#dataを2バイトの数値列に変換
data=np.frombuffer(data, dtype=np.int16)

#dataの値を2バイトの変数が取り得る値の最大値で正規化
data=data/np.iinfo(np.int16).max

#waveファイルを閉じる
wav.close()

#x軸の値
x=np.array(range(wav.getnframes()))/wav.getframerate()

#音声データをプロットする
plt.figure(figsize=(10,4))

#x軸のラベル
plt.xlabel("Time [sec]")

#y軸のラベル
plt.ylabel("Value [-1,1]")

#データをプロット
plt.plot(x,data)

#音声ファイルを画像として保存
plt.savefig("./wave_form.png")

#画像を画面に表示
plt.show()

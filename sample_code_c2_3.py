

import wave as wave
import numpy as np
import matplotlib.pyplot as plt

#白色雑音のサンプル数を設定
n_sample=40000

#サンプリング周波数
sample_rate=16000

#乱数の種を設定
np.random.seed(0)

#白色雑音を生成
data=np.random.normal(size=n_sample)

#x軸の値
x=np.array(range(n_sample))/sample_rate

#音声データをプロットする
plt.figure(figsize=(10,4))

#x軸のラベル
plt.xlabel("Time [sec]")

#y軸のラベル
plt.ylabel("Value")

#データをプロット
plt.plot(x,data)

#音声ファイルを画像として保存
plt.savefig("./wgn_wave_form.png")

#画像を画面に表示
plt.show()

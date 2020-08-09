

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
data=np.random.normal(scale=0.1,size=n_sample)

#2バイトのデータとして書き込むためにスケールを調整 
data_scale_adjust=data*np.iinfo(np.int16).max

#2バイトのデータに変換
data_scale_adjust=data_scale_adjust.astype(np.int16)


#waveファイルに書き込む
wave_out = wave.open("./wgn_wave.wav", 'w')

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



import wave as wave
import pyroomacoustics as pa
import numpy as np
import matplotlib.pyplot as plt

#乱数の種を初期化
np.random.seed(0)

#畳み込みに用いる音声波形
clean_wave_files=["./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav","./CMU_ARCTIC/cmu_us_axb_arctic/wav/arctic_a0002.wav"]

#音源数
n_sources=len(clean_wave_files)

#長さを調べる
n_samples=0
#ファイルを読み込む
for clean_wave_file in clean_wave_files:
    wav=wave.open(clean_wave_file)
    if n_samples<wav.getnframes():
        n_samples=wav.getnframes()
    wav.close()

clean_data=np.zeros([n_sources,n_samples])

#ファイルを読み込む
s=0
for clean_wave_file in clean_wave_files:
    wav=wave.open(clean_wave_file)
    data=wav.readframes(wav.getnframes())
    data=np.frombuffer(data, dtype=np.int16)
    data=data/np.iinfo(np.int16).max
    clean_data[s,:wav.getnframes()]=data
    wav.close()
    s=s+1
# シミュレーションのパラメータ
#サンプリング周波数
sample_rate=16000

#音声と雑音との比率 [dB]
SNR=90.

#部屋の大きさ
room_dim = np.r_[10.0, 10.0, 10.0]

#マイクロホンアレイを置く部屋の場所
mic_array_loc = room_dim / 2 + np.random.randn(3) * 0.1 

#マイクロホンアレイのマイク配置
mic_alignments = np.array(
    [
        [-0.01, 0.0, 0.0],
        [0.01, 0.0, 0.0],
    ]
)

#マイクロホン数
n_channels=np.shape(mic_alignments)[0]

#マイクロホンアレイの座標
R=mic_alignments .T+mic_array_loc[:,None]

# 部屋を生成する
room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=17,absorption=0.35)

# 用いるマイクロホンアレイの情報を設定する
room.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))

#音源の場所
doas=np.array(
    [[np.pi/2., 0],
     [np.pi/2.,np.pi/2.]
    ]    )

#音源とマイクロホンの距離
distance=1.
source_locations=np.zeros((3, doas.shape[0]), dtype=doas.dtype)
source_locations[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[2, :] = np.cos(doas[:, 0])
source_locations *= distance
source_locations += mic_array_loc[:, None]

#各音源をシミュレーションに追加する
for s in range(n_sources):
    clean_data[s]/= np.std(clean_data[s])
    room.add_source(source_locations[:, s], signal=clean_data[s])

#シミュレーションを回す
room.simulate(snr=SNR)

#インパルス応答を取得する
#room.rirにはマイク，音源の順番で各音源の各マイクのインパルス応答が入っている
impulse_responses=room.rir

impulse_length=np.shape(impulse_responses[0][0])[0]

#残響時間を取得
rt60=pa.experimental.measure_rt60(impulse_responses[0][0],fs=sample_rate)
print("残響時間:{} [sec]".format(rt60))

rir_power=np.square(impulse_responses[0][0])


reverb_power=np.zeros_like(rir_power)
for t in range(impulse_length):
    reverb_power[t]=10.*np.log10(np.sum(rir_power[t:])/np.sum(rir_power))
    


#x軸の値
x=np.array(range(impulse_length))/sample_rate

#音声データをプロットする
plt.figure(figsize=(10,4))

#x軸のラベル
plt.xlabel("Time [sec]")

#y軸のラベル
plt.ylabel("Value")

#x軸の範囲を設定する
plt.xlim([0,0.5])

#データをプロット
plt.plot(x,impulse_responses[0][0])
#plt.plot(x,reverb_power)

#音声ファイルを画像として保存
plt.savefig("./impulse_responses2.png")

#画像を画面に表示
plt.show()





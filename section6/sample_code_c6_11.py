

import wave as wave
import pyroomacoustics as pa
import numpy as np
import scipy.signal as sp
import scipy as scipy

#ステアリングベクトルを算出
#mic_position: 3 x M  dimensional ndarray [[x,y,z],[x,y,z]]
#source_position: 3x Ns dimensional ndarray [[x,y,z],[x,y,z] ]
#freqs: Nk dimensional array [f1,f2,f3...]
#sound_speed: 音速 [m/s]
#is_use_far: Farを使う場合はTrue, Nearの場合はFalse, 
#return: steering vector (Nk x Ns x M)
def calculate_steering_vector(mic_alignments,source_locations,freqs,sound_speed=340,is_use_far=False):
    #マイク数を取得
    n_channels=np.shape(mic_alignments)[1]

    #音源数を取得
    n_source=np.shape(source_locations)[1]

    if is_use_far==True:
        #音源位置を正規化
        norm_source_locations=source_locations/np.linalg.norm(source_locations,2,axis=0,keepdims=True)

        #位相を求める
        steering_phase=np.einsum('k,ism,ism->ksm',2.j*np.pi/sound_speed*freqs,norm_source_locations[...,None],mic_alignments[:,None,:])

        #ステアリングベクトルを算出
        steering_vector=1./np.sqrt(n_channels)*np.exp(steering_phase)

        return(steering_vector)

    else:

        #音源とマイクの距離を求める
        #distance: Ns x Nm
        distance=np.sqrt(np.sum(np.square(source_locations[...,None]-mic_alignments[:,None,:]),axis=0))

        #遅延時間(delay) [sec]
        delay=distance/sound_speed

        #ステアリングベクトルの位相を求める
        steering_phase=np.einsum('k,sm->ksm',-2.j*np.pi*freqs,delay)
    
        #音量の減衰
        steering_decay_ratio=1./distance

        #ステアリングベクトルを求める
        steering_vector=steering_decay_ratio[None,...]*np.exp(steering_phase)

        #大きさを1で正規化する
        steering_vector=steering_vector/np.linalg.norm(steering_vector,2,axis=2,keepdims=True)

    return(steering_vector)

#2バイトに変換してファイルに保存
#signal: time-domain 1d array (float)
#file_name: 出力先のファイル名
#sample_rate: サンプリングレート
def write_file_from_time_signal(signal,file_name,sample_rate):
    #2バイトのデータに変換
    signal=signal.astype(np.int16)

    #waveファイルに書き込む
    wave_out = wave.open(file_name, 'w')

    #モノラル:1、ステレオ:2
    wave_out.setnchannels(1)

    #サンプルサイズ2byte
    wave_out.setsampwidth(2)

    #サンプリング周波数
    wave_out.setframerate(sample_rate)

    #データを書き込み
    wave_out.writeframes(signal)

    #ファイルを閉じる
    wave_out.close()

#乱数の種を初期化
np.random.seed(0)

#畳み込みに用いる音声波形
clean_wave_files=["./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav"]


#雑音だけの区間のサンプル数を設定
n_noise_only=40000

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

clean_data=np.zeros([n_sources,n_samples+n_noise_only])

#ファイルを読み込む
s=0
for clean_wave_file in clean_wave_files:
    wav=wave.open(clean_wave_file)
    data=wav.readframes(wav.getnframes())
    data=np.frombuffer(data, dtype=np.int16)
    data=data/np.iinfo(np.int16).max
    clean_data[s,n_noise_only:n_noise_only+wav.getnframes()]=data
    wav.close()
    s=s+1
# シミュレーションのパラメータ

#サンプリング周波数
sample_rate=16000

#フレームサイズ
N=1024

#周波数の数
Nk=int(N/2+1)


#各ビンの周波数
freqs=np.arange(0,Nk,1)*sample_rate/N

#音声と雑音との比率 [dB]
SNR=20.

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
room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)

# 用いるマイクロホンアレイの情報を設定する
room.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))

#音源の場所
doas=np.array(
    [[np.pi/2., 0]
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

#畳み込んだ波形を取得する(チャンネル、サンプル）
multi_conv_data=room.mic_array.signals


#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data[0,n_noise_only:]*np.iinfo(np.int16).max/20.,"./maxsnr_in.wav",sample_rate)

#Near仮定に基づくステアリングベクトルを計算: steering_vectors(Nk x Ns x M)
near_steering_vectors=calculate_steering_vector(R,source_locations,freqs,is_use_far=False)

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(multi_conv_data,fs=sample_rate,window="hann",nperseg=N)

#雑音だけの区間のフレーム数
n_noise_only_frame=np.sum(t<(n_noise_only/sample_rate))

xx_H=np.einsum("mkt,nkt->ktmn",stft_data,np.conjugate(stft_data))

#雑音の共分散行列 freq,mic,mic
Rn=np.average(xx_H[:,:n_noise_only_frame,...],axis=1)

#入力共分散行列
Rs=np.average(xx_H[:,n_noise_only_frame:,...],axis=1)

#一般化固有値分解
max_snr_filter=None
for k in range(Nk):
    w,v=scipy.linalg.eigh(Rs[k,...],Rn[k,...])
    if max_snr_filter is None:
        max_snr_filter=v[None,:,-1]
    else:
        max_snr_filter=np.concatenate((max_snr_filter,v[None,:,-1]),axis=0)
    

Rs_w=np.einsum("kmn,kn->km",Rs,max_snr_filter)
beta=Rs_w[:,0]/np.einsum("km,km->k",np.conjugate(max_snr_filter),Rs_w)
w_max_snr=beta[:,None]*max_snr_filter

#フィルタをかける
c_hat=np.einsum("km,mkt->kt",np.conjugate(w_max_snr),stft_data)

#時間領域の波形に戻す
t,maxsnr_out=sp.istft(c_hat,fs=sample_rate,window="hann",nperseg=N)

#大きさを調整する
maxsnr_out=maxsnr_out*np.iinfo(np.int16).max/20.

#ファイルに書き込む
write_file_from_time_signal(maxsnr_out[n_noise_only:],"./maxsnr_out.wav",sample_rate)

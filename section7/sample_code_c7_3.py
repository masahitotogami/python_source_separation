

import wave as wave
import pyroomacoustics as pa
import numpy as np
import scipy.signal as sp
import scipy as scipy



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


#スパース性に基づく分離
#input_vectors:マイクロホン入力信号(Nm,Nk, Lt)
#steering_vectors:ステアリングベクトル(Nk,|Ω|,Nm)
#omega: 目的音の範囲(Ns,|Ω|) 
#return y：出力信号(Nm,Ns,Nk, Lt)
def execute_doa_sparse_separation(input_vectors,steering_vectors,omega):
    inner_product=np.einsum("kim,mkt->kit",np.conjugate(steering_vectors),input_vectors)
    
    n_omega=np.shape(omega)[1]

    estimate_doas=np.argmax(np.abs(inner_product),axis=1)

    estimate_doa_mask=np.identity(n_omega)[estimate_doas]

    output_mask=np.einsum("kti,si->skt",estimate_doa_mask,omega)

    y=np.einsum("skt,mkt->mskt",output_mask,input_vectors)

    return(y)


#SNRをはかる
#desired: 目的音、Lt
#out:　雑音除去後の信号 Lt
def calculate_snr(desired,out):
    wave_length=np.minimum(np.shape(desired)[0],np.shape(out)[0])

    #消し残った雑音
    desired=desired[:wave_length]
    out=out[:wave_length]
    noise=desired-out
    snr=10.*np.log10(np.sum(np.square(desired))/np.sum(np.square(noise)))

    return(snr)

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

#シミュレーションで用いる音源数
n_sim_sources=2

#サンプリング周波数
sample_rate=16000

#フレームサイズ
N=1024

#周波数の数
Nk=int(N/2+1)

#各ビンの周波数
freqs=np.arange(0,Nk,1)*sample_rate/N

#音声と雑音との比率 [dB]
SNR=90.

#方位角の閾値
azimuth_th=30.

#部屋の大きさ
room_dim = np.r_[10.0, 10.0, 10.0]

#マイクロホンアレイを置く部屋の場所
mic_array_loc = room_dim / 2 + np.random.randn(3) * 0.1 

#マイクロホンアレイのマイク配置
mic_directions=np.array(
    [[np.pi/2., theta/180.*np.pi] for theta in np.arange(0,360,10)
    ]    )

distance=0.02
mic_alignments=np.zeros((3, mic_directions.shape[0]), dtype=mic_directions.dtype)
mic_alignments[0, :] = np.cos(mic_directions[:, 1]) * np.sin(mic_directions[:, 0])
mic_alignments[1, :] = np.sin(mic_directions[:, 1]) * np.sin(mic_directions[:, 0])
mic_alignments[2, :] = np.cos(mic_directions[:, 0])
mic_alignments *= distance

#マイクロホン数
n_channels=np.shape(mic_alignments)[1]

#マイクロホンアレイの座標
R=mic_alignments+mic_array_loc[:,None]

# 部屋を生成する
room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)
room_no_noise_left = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)
room_no_noise_right = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)

# 用いるマイクロホンアレイの情報を設定する
room.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
room_no_noise_left.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
room_no_noise_right.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))

#音源の場所
doas=np.array(
    [[np.pi/2., np.pi],
     [np.pi/2., 0]
    ]    )

#音源とマイクロホンの距離
distance=1.
source_locations=np.zeros((3, doas.shape[0]), dtype=doas.dtype)
source_locations[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[2, :] = np.cos(doas[:, 0])
source_locations *= distance
source_locations += mic_array_loc[:, None]


#ステアリングベクトルを算出するための仮想的な音源方向
virtual_doas=np.array(
    [[np.pi/2., theta/180.*np.pi] for theta in np.arange(0,360,5)
    ]    )
virtual_source_locations=np.zeros((3, virtual_doas.shape[0]), dtype=virtual_doas.dtype)
virtual_source_locations[0, :] = np.cos(virtual_doas[:, 1]) * np.sin(virtual_doas[:, 0])
virtual_source_locations[1, :] = np.sin(virtual_doas[:, 1]) * np.sin(virtual_doas[:, 0])
virtual_source_locations[2, :] = np.cos(virtual_doas[:, 0])
virtual_source_locations *= 100.
virtual_source_locations += mic_array_loc[:, None]

#仮想的な音源方向のステアリングベクトル作成 
#virtual steering vector: (Nk x Ns x M)
virtual_steering_vectors=calculate_steering_vector(R,virtual_source_locations,freqs,is_use_far=True)

def modify_angle_diff(diff):
    diff=np.where(diff<-np.pi,diff+np.pi*2,diff)
    diff=np.where(diff>np.pi,diff-np.pi*2,diff)
    return(diff)

#所望音の方向から±th度以内
omega=np.array([ np.abs(modify_angle_diff(virtual_doas[:,1]-doas[s,1]))<azimuth_th/180.*np.pi   for s in range(n_sim_sources)]).astype(np.float)

#各音源をシミュレーションに追加する
for s in range(n_sim_sources):
    clean_data[s]/= np.std(clean_data[s])
    room.add_source(source_locations[:, s], signal=clean_data[s])
    if s==0:
        room_no_noise_left.add_source(source_locations[:, s], signal=clean_data[s])
    if s==1:
        room_no_noise_right.add_source(source_locations[:, s], signal=clean_data[s])

#シミュレーションを回す
room.simulate(snr=SNR)
room_no_noise_left.simulate(snr=90)
room_no_noise_right.simulate(snr=90)

#畳み込んだ波形を取得する(チャンネル、サンプル）
multi_conv_data=room.mic_array.signals
multi_conv_data_left_no_noise=room_no_noise_left.mic_array.signals
multi_conv_data_right_no_noise=room_no_noise_right.mic_array.signals




#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data_left_no_noise[0,:]*np.iinfo(np.int16).max/20.,"./doa_left_clean.wav",sample_rate)

#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data_right_no_noise[0,:]*np.iinfo(np.int16).max/20.,"./doa_right_clean.wav",sample_rate)

#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data[0,:]*np.iinfo(np.int16).max/20.,"./doa_in_left.wav",sample_rate)
write_file_from_time_signal(multi_conv_data[0,:]*np.iinfo(np.int16).max/20.,"./doa_in_right.wav",sample_rate)

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(multi_conv_data,fs=sample_rate,window="hann",nperseg=N)

#DOA情報を使って分離
y_doa=execute_doa_sparse_separation(stft_data,virtual_steering_vectors,omega)


#時間領域の波形に戻す
t,y_doa_left=sp.istft(y_doa[0,0,...],fs=sample_rate,window="hann",nperseg=N)
t,y_doa_right=sp.istft(y_doa[0,1,...],fs=sample_rate,window="hann",nperseg=N)


#SNRを計算
snr_pre=calculate_snr(multi_conv_data_left_no_noise[0,...],multi_conv_data[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],multi_conv_data[0,...])
snr_doa_post=calculate_snr(multi_conv_data_left_no_noise[0,...],y_doa_left)+calculate_snr(multi_conv_data_right_no_noise[0,...],y_doa_right)
snr_pre/=2.
snr_doa_post/=2.

#ファイルに書き込む
write_file_from_time_signal(y_doa_left*np.iinfo(np.int16).max/20.,"./sparse_doa_left.wav",sample_rate)
write_file_from_time_signal(y_doa_right*np.iinfo(np.int16).max/20.,"./sparse_doa_right.wav",sample_rate)


print("method:    ", "DOA")

print("Δsnr [dB]: {:.2f} ".format(snr_doa_post-snr_pre))

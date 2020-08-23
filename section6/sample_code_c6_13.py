

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

#遅延和アレイを実行する
#x:入力信号( M, Nk, Lt)
#a:ステアリングベクトル(Nk, M)
#return y 出力信号(M, Nk, Lt)
def execute_dsbf(x,a):
    #遅延和アレイを実行する
    s_hat=np.einsum("km,mkt->kt",np.conjugate(a),x)

    #ステアリングベクトルをかける
    c_hat=np.einsum("kt,km->mkt",s_hat,a)

    return(c_hat)

#MVDRを実行する
#x:入力信号( M, Nk, Lt)
#y:共分散行列を計算する信号(M,Nk,Lt)
#a:ステアリングベクトル(Nk, M)
#return y 出力信号(M, Nk, Lt)
def execute_mvdr(x,y,a):
    
    #共分散行列を計算する
    Rcov=np.einsum("mkt,nkt->kmn",y,np.conjugate(y))

    #共分散行列の逆行列を計算する
    Rcov_inverse=np.linalg.pinv(Rcov)

    #フィルタを計算する
    Rcov_inverse_a=np.einsum("kmn,kn->km",Rcov_inverse,a)
    a_H_Rcov_inverse_a=np.einsum("kn,kn->k",np.conjugate(a),Rcov_inverse_a)
    w_mvdr=Rcov_inverse_a/np.maximum(a_H_Rcov_inverse_a,1.e-18)[:,None]

    #フィルタをかける
    s_hat=np.einsum("km,mkt->kt",np.conjugate(w_mvdr),x)

    #ステアリングベクトルをかける
    c_hat=np.einsum("kt,km->mkt",s_hat,a)
    
    return(c_hat)

#MaxSNRを実行する
#x:入力信号( M, Nk, Lt)
#y:共分散行列を計算する信号(M,Nk,Lt)
#return y 出力信号(M, Nk, Lt)
def execute_max_snr(x,y):
   
    #雑音の共分散行列 freq,mic,mic
    Rn=np.average(np.einsum("mkt,nkt->ktmn",y,np.conjugate(y)),axis=1)

    #入力共分散行列
    Rs=np.average(np.einsum("mkt,nkt->ktmn",x,np.conjugate(x)),axis=1)

    #周波数の数を取得
    Nk=np.shape(Rs)[0]

    #一般化固有値分解
    max_snr_filter=None
    for k in range(Nk):
        w,v=scipy.linalg.eigh(Rs[k,...],Rn[k,...])
        if max_snr_filter is None:
            max_snr_filter=v[None,:,-1]
        else:
            max_snr_filter=np.concatenate((max_snr_filter,v[None,:,-1]),axis=0)
    

    Rs_w=np.einsum("kmn,kn->km",Rs,max_snr_filter)
    beta=Rs_w/np.einsum("km,km->k",np.conjugate(max_snr_filter),Rs_w)[:,None]
    w_max_snr=beta[:,None,:]*max_snr_filter[...,None]
    
    #フィルタをかける
    c_hat=np.einsum("kim,ikt->mkt",np.conjugate(w_max_snr),x)
    
    return(c_hat)

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

#ファイルに書き込む
#MWFを実行する
#x:入力信号( M, Nk, Lt)
#y:共分散行列を計算する信号(M,Nk,Lt)
#mu: 雑音共分散行列の係数
#return y 出力信号(M, Nk, Lt)
def execute_mwf(x,y,mu):
   
    #雑音の共分散行列 freq,mic,mic
    Rn=np.average(np.einsum("mkt,nkt->ktmn",y,np.conjugate(y)),axis=1)

    #入力共分散行列
    Rs=np.average(np.einsum("mkt,nkt->ktmn",x,np.conjugate(x)),axis=1)
    
    #固有値分解をして半正定行列に変換
    w,v=np.linalg.eigh(Rs)
    Rs_org=Rs.copy()
    w[np.real(w)<0]=0
    Rs=np.einsum("kmi,ki,kni->kmn",v,w,np.conjugate(v))

    #入力共分散行列
    Rs_muRn=Rs+Rn*mu
    invRs_muRn=np.linalg.pinv(Rs_muRn)

    #フィルタ生成
    W_mwf=np.einsum("kmi,kin->kmn",invRs_muRn,Rs)

    #フィルタをかける
    c_hat=np.einsum("kim,ikt->mkt",np.conjugate(W_mwf),x)

    return(c_hat)

#乱数の種を初期化
np.random.seed(0)

#畳み込みに用いる音声波形
clean_wave_files=["./CMU_ARCTIC/cmu_us_aew_arctic/wav/arctic_a0001.wav","./CMU_ARCTIC/cmu_us_axb_arctic/wav/arctic_a0002.wav"]

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

#シミュレーションで用いる音源数
n_sim_sources=1

#サンプリング周波数
sample_rate=16000

#フレームサイズ
N=1024

#周波数の数
Nk=int(N/2+1)


#各ビンの周波数
freqs=np.arange(0,Nk,1)*sample_rate/N

#音声と雑音との比率 [dB]
SNR=10.

#部屋の大きさ
room_dim = np.r_[10.0, 10.0, 10.0]

#マイクロホンアレイを置く部屋の場所
mic_array_loc = room_dim / 2 + np.random.randn(3) * 0.1 

#マイクロホンアレイのマイク配置
mic_alignments = np.array(
    [
        [x, 0.0, 0.0] for x in np.arange(-0.01,0.02,0.02)
    ]
)

#マイクロホン数
n_channels=np.shape(mic_alignments)[0]

#マイクロホンアレイの座標
R=mic_alignments .T+mic_array_loc[:,None]

# 部屋を生成する
room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=17,absorption=0.4)
room_no_noise = pa.ShoeBox(room_dim, fs=sample_rate, max_order=17,absorption=0.4)

# 用いるマイクロホンアレイの情報を設定する
room.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))
room_no_noise.add_microphone_array(pa.MicrophoneArray(R, fs=room.fs))

#音源の場所
doas=np.array(
    [[np.pi/2., 0],
     [np.pi/2., np.pi]
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
for s in range(n_sim_sources):
    clean_data[s]/= np.std(clean_data[s])
    room.add_source(source_locations[:, s], signal=clean_data[s])
    if s==0:
        room_no_noise.add_source(source_locations[:, s], signal=clean_data[s])

#シミュレーションを回す
room.simulate(snr=SNR)
room_no_noise.simulate(snr=90)

#畳み込んだ波形を取得する(チャンネル、サンプル）
multi_conv_data=room.mic_array.signals
multi_conv_data_no_noise=room_no_noise.mic_array.signals



#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data_no_noise[0,n_noise_only:]*np.iinfo(np.int16).max/20.,"./mwf_clean.wav",sample_rate)

#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data[0,n_noise_only:]*np.iinfo(np.int16).max/20.,"./mwf_in.wav",sample_rate)

#Near仮定に基づくステアリングベクトルを計算: steering_vectors(Nk x Ns x M)
near_steering_vectors=calculate_steering_vector(R,source_locations,freqs,is_use_far=False)

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(multi_conv_data,fs=sample_rate,window="hann",nperseg=N)

#雑音だけの区間のフレーム数
n_noise_only_frame=np.sum(t<(n_noise_only/sample_rate))

#雑音だけのデータ
noise_data=stft_data[...,:n_noise_only_frame]

#MWFの雑音の倍率
mu=1.0

#それぞれのフィルタを実行する
dsbf_out=execute_dsbf(stft_data,near_steering_vectors[:,0,:])
mvdr_out=execute_mvdr(stft_data,stft_data,near_steering_vectors[:,0,:])
mlbf_out=execute_mvdr(stft_data,noise_data,near_steering_vectors[:,0,:])
max_snr_out=execute_max_snr(stft_data,noise_data)
mwf_out=execute_mwf(stft_data,noise_data,mu)

#評価するマイクロホン
eval_mic_index=0

#時間領域の波形に戻す
t,dsbf_out=sp.istft(dsbf_out[eval_mic_index],fs=sample_rate,window="hann",nperseg=N)
t,mvdr_out=sp.istft(mvdr_out[eval_mic_index],fs=sample_rate,window="hann",nperseg=N)
t,mlbf_out=sp.istft(mlbf_out[eval_mic_index],fs=sample_rate,window="hann",nperseg=N)
t,max_snr_out=sp.istft(max_snr_out[eval_mic_index],fs=sample_rate,window="hann",nperseg=N)
t,mwf_out=sp.istft(mwf_out[eval_mic_index],fs=sample_rate,window="hann",nperseg=N)

#SNRをはかる
snr_pre=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],multi_conv_data[eval_mic_index,n_noise_only:])
snr_dsbf_post=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],dsbf_out[n_noise_only:])
snr_mvdr_post=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],mvdr_out[n_noise_only:])
snr_mlbf_post=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],mlbf_out[n_noise_only:])
snr_max_snr_post=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],max_snr_out[n_noise_only:])
snr_mwf_post=calculate_snr(multi_conv_data_no_noise[eval_mic_index,n_noise_only:],mwf_out[n_noise_only:])

#ファイルに書き込む
write_file_from_time_signal(multi_conv_data[eval_mic_index,n_noise_only:]*np.iinfo(np.int16).max/20.,"./mix.wav",sample_rate)
write_file_from_time_signal(multi_conv_data_no_noise[eval_mic_index,n_noise_only:]*np.iinfo(np.int16).max/20.,"./desired.wav",sample_rate)
write_file_from_time_signal(dsbf_out[n_noise_only:]*np.iinfo(np.int16).max/20.,"./dsbf_out.wav",sample_rate)
write_file_from_time_signal(mvdr_out[n_noise_only:]*np.iinfo(np.int16).max/20.,"./mvdr_out.wav",sample_rate)
write_file_from_time_signal(mlbf_out[n_noise_only:]*np.iinfo(np.int16).max/20.,"./mlbf_out.wav",sample_rate)
write_file_from_time_signal(max_snr_out[n_noise_only:]*np.iinfo(np.int16).max/20.,"./max_snr_out.wav",sample_rate)
write_file_from_time_signal(mwf_out[n_noise_only:]*np.iinfo(np.int16).max/20.,"./mwf_out.wav",sample_rate)


print("method:    ", "DSBF", "MVDR", "MLBF", "MaxSNR", "MWF")

print("Δsnr [dB]: {:.2f} {:.2f} {:.2f} {:.2f}   {:.2f}".format(snr_dsbf_post-snr_pre,snr_mvdr_post-snr_pre,snr_mlbf_post-snr_pre,snr_max_snr_post-snr_pre,snr_mwf_post-snr_pre))

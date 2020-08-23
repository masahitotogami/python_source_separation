

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

#共分散行列からステアリングベクトルを推定する
#Rs: 共分散行列(Ns, Nk, M, M)
#steering_vector(Ns, Nk, M)
def estimate_steering_vector(Rs):

    #固有値分解を実施して最大固有値を与える固有ベクトルを取得
    w,v=np.linalg.eigh(Rs)
    steering_vector=v[...,-1]
    return(steering_vector)

#マスクと入力信号から共分散行列を推定
#x:入力信号(M,Nk,Lt)
#mask：音源毎の時間周波数マスク(Ns, Nk, Lt)
#return Rs 音源共分散行列(Ns, Nk,M,M), Rn 雑音共分散行列(Ns, Nk,M,M)
def estimate_covariance_matrix(x,mask):
    
    #目的音の共分散行列を推定する
    Rs=np.einsum("skt,mkt,nkt->skmn",mask,x,np.conjugate(x))
    sum_mask=np.sum(mask,axis=2)
    Rs=Rs/np.maximum(sum_mask,1.e-18)[...,None,None]

    #雑音共分散行列を推定する
    Rn=np.einsum("skt,mkt,nkt->skmn",1.-mask,x,np.conjugate(x))
    sum_noise_mask=np.sum(1.-mask,axis=2)
    Rn=Rn/np.maximum(sum_noise_mask,1.e-18)[...,None,None]

    
    #固有値分解をして半正定行列に変換
    w,v=np.linalg.eigh(Rs)
    Rs_org=Rs.copy()
    w[np.real(w)<1.e-18]=1.e-18
    Rs=np.einsum("skmi,ski,skni->skmn",v,w,np.conjugate(v))
    w,v=np.linalg.eigh(Rn)
    Rn_org=Rn.copy()
    w[np.real(w)<1.e-18]=1.e-18
    Rn=np.einsum("skmi,ski,skni->skmn",v,w,np.conjugate(v))

    return(Rs,Rn)

#スパース性に基づく分離
#x:入力信号(M,Nk,Lt)
#mask：音源毎の時間周波数マスク(Ns, Nk, Lt)
#return c_hat：出力信号(Nm,Ns,Nk, Lt)
def execute_sparse(x,mask):


    c_hat=np.einsum("skt,mkt->mskt",mask,x)

    return(c_hat)

#ファイルに書き込む
#MWFを実行する
#x:入力信号( M, Nk, Lt)
#Rn: 雑音共分散行列(Ns, Nk,M,M)
#Rs: 音源共分散行列(Ns, Nk,M,M)
#return c_hat 出力信号(M, Ns,Nk, Lt)
def execute_mwf(x,Rn,Rs):
    
    #入力信号に対する共分散行列の逆行列を計算
    Rx_inverse=np.linalg.pinv(Rn+Rs)
    
    #フィルタ生成
    W_mwf=np.einsum("skmi,skin->skmn",Rx_inverse,Rs)
    
    #フィルタをかける
    c_hat=np.einsum("skim,ikt->mskt",np.conjugate(W_mwf),x)

    return(c_hat)

#遅延和アレイを実行する
#x:入力信号( M, Nk, Lt)
#a:ステアリングベクトル(Ns,Nk, M)
#return c_hat 出力信号(M, Ns,Nk, Lt)
def execute_dsbf(x,a):

    #遅延和アレイを実行する
    s_hat=np.einsum("skm,mkt->skt",np.conjugate(a),x)

    #ステアリングベクトルをかける
    c_hat=np.einsum("skt,skm->mskt",s_hat,a)

    return(c_hat)

#MVDRを実行する
#x:入力信号( M, Nk, Lt)
#Rn: 雑音共分散行列(Ns, Nk,M,M)
#a:ステアリングベクトル(Ns,Nk, M)
#return c_hat 出力信号(M, Ns,Nk, Lt)
def execute_mvdr(x,Rn,a):

    #共分散行列の逆行列を計算する
    Rn_inverse=np.linalg.pinv(Rn)

    #フィルタを計算する
    Rn_inverse_a=np.einsum("skmn,skn->skm",Rn_inverse,a)
    a_H_Rn_inverse_a=np.einsum("skn,skn->sk",np.conjugate(a),Rn_inverse_a)
    w_mvdr=Rn_inverse_a/np.maximum(a_H_Rn_inverse_a,1.e-18)[...,None]

    #フィルタをかける
    s_hat=np.einsum("skm,mkt->skt",np.conjugate(w_mvdr),x)

    #ステアリングベクトルをかける
    c_hat=np.einsum("skt,skm->mskt",s_hat,a)
    
    return(c_hat)

#MVDR2を実行する(共分散行列のみから計算)
#x:入力信号( M, Nk, Lt)
#Rn: 雑音共分散行列(Ns, Nk,M,M)
#Rs:目的音の共分散行列(Ns,Nk, M,M)
#return c_hat 出力信号(M, Ns,Nk, Lt)
def execute_mvdr2(x,Rn,Rs):

    #共分散行列の逆行列を計算する
    Rn_inverse=np.linalg.pinv(Rn)

    #フィルタを計算する
    Rn_inverse_Rs=np.einsum("skmi,skin->skmn",Rn_inverse,Rs)
    
    w_mvdr=Rn_inverse_Rs/np.maximum(np.trace(Rn_inverse_Rs,axis1=-2,axis2=-1),1.e-18)[...,None,None]

    #フィルタをかける
    c_hat=np.einsum("skmn,mkt->nskt",np.conjugate(w_mvdr),x)

    return(c_hat)

#MaxSNRを実行する
#x:入力信号( M, Nk, Lt)
#Rn: 雑音共分散行列(Ns, Nk,M,M)
#Rs:目的音の共分散行列(Ns,Nk, M,M)
#return c_hat 出力信号(M, Ns,Nk, Lt)
def execute_max_snr(x,Rn,Rs):
   
    #音源・周波数・マイクロホンの数を取得
    Ns=np.shape(Rs)[0]
    Nk=np.shape(Rs)[1]
    M=np.shape(Rs)[2]

    #一般化固有値分解
    max_snr_filter=None
    for s in range(Ns):
        for k in range(Nk):
            w,v=scipy.linalg.eigh(Rs[s,k,...],Rn[s,k,...])
            if max_snr_filter is None:
                max_snr_filter=v[None,:,-1]
            else:
                max_snr_filter=np.concatenate((max_snr_filter,v[None,:,-1]),axis=0)
    max_snr_filter=np.reshape(max_snr_filter,(Ns,Nk,M))

    Rs_w=np.einsum("skmn,skn->skm",Rs,max_snr_filter)
    beta=Rs_w/np.einsum("skm,skm->sk",np.conjugate(max_snr_filter),Rs_w)[...,None]
    w_max_snr=beta[...,None,:]*max_snr_filter[...,None]
    
    #フィルタをかける
    c_hat=np.einsum("skim,ikt->mskt",np.conjugate(w_max_snr),x)
    
    return(c_hat)

#時間周波数マスクを推定する
#input_vectors:マイクロホン入力信号(Nm,Nk, Lt)
#steering_vectors:ステアリングベクトル(Nk,|Ω|,Nm)
#omega: 目的音の範囲(Ns,|Ω|) 
#return mask：出力信号(Ns,Nk, Lt)
def estimate_mask(input_vectors,steering_vectors,omega):
    inner_product=np.einsum("kim,mkt->kit",np.conjugate(steering_vectors),input_vectors)
    
    n_omega=np.shape(omega)[1]

    estimate_doas=np.argmax(np.abs(inner_product),axis=1)

    estimate_doa_mask=np.identity(n_omega)[estimate_doas]

    mask=np.einsum("kti,si->skt",estimate_doa_mask,omega)

    return(mask)


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
    [[np.pi/2., theta/180.*np.pi] for theta in np.arange(180,361,180)
    ]    )

distance=0.01
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

#音源の誤差
doas_error=np.array(
    [[0., 5./180.*np.pi],
     [0., 5./180.*np.pi]
    ]    )
#音源とマイクロホンの距離
distance=1.
doas2=doas+doas_error

source_locations=np.zeros((3, doas2.shape[0]), dtype=doas2.dtype)
source_locations[0, :] = np.cos(doas2[:, 1]) * np.sin(doas2[:, 0])
source_locations[1, :] = np.sin(doas2[:, 1]) * np.sin(doas2[:, 0])
source_locations[2, :] = np.cos(doas2[:, 0])
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
write_file_from_time_signal(multi_conv_data_left_no_noise[0,:]*np.iinfo(np.int16).max/20.,"./doa_bf_left_clean.wav",sample_rate)

#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data_right_no_noise[0,:]*np.iinfo(np.int16).max/20.,"./doa_bf_right_clean.wav",sample_rate)

#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data[0,:]*np.iinfo(np.int16).max/20.,"./doa_bf_in_left.wav",sample_rate)
write_file_from_time_signal(multi_conv_data[0,:]*np.iinfo(np.int16).max/20.,"./doa_bf_in_right.wav",sample_rate)

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(multi_conv_data,fs=sample_rate,window="hann",nperseg=N)

#時間周波数マスクを推定する
tf_mask=estimate_mask(stft_data,virtual_steering_vectors,omega)

#共分散行列とステアリングベクトルを推定
Rs,Rn=estimate_covariance_matrix(stft_data,tf_mask)
desired_steering_vectors=estimate_steering_vector(Rs)

#各フィルタを実行する
y_sparse=execute_sparse(stft_data,tf_mask)
y_dsbf=execute_dsbf(stft_data,desired_steering_vectors)
y_mvdr=execute_mvdr(stft_data,Rn,desired_steering_vectors)
y_mvdr2=execute_mvdr2(stft_data,Rn,Rs)
y_max_snr=execute_max_snr(stft_data,Rn,Rs)
y_mwf=execute_mwf(stft_data,Rn,Rs)



#時間領域の波形に戻す
t,y_sparse=sp.istft(y_sparse[0,...],fs=sample_rate,window="hann",nperseg=N)
t,y_dsbf=sp.istft(y_dsbf[0,...],fs=sample_rate,window="hann",nperseg=N)
t,y_mvdr=sp.istft(y_mvdr[0,...],fs=sample_rate,window="hann",nperseg=N)
t,y_mvdr2=sp.istft(y_mvdr2[0,...],fs=sample_rate,window="hann",nperseg=N)
t,y_max_snr=sp.istft(y_max_snr[0,...],fs=sample_rate,window="hann",nperseg=N)
t,y_mwf=sp.istft(y_mwf[0,...],fs=sample_rate,window="hann",nperseg=N)


#SNRをはかる
snr_pre=calculate_snr(multi_conv_data_left_no_noise[0,...],multi_conv_data[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],multi_conv_data[0,...])
snr_doa_sparse_post=calculate_snr(multi_conv_data_left_no_noise[0,...],y_sparse[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_sparse[1,...])
snr_doa_dsbf_post=calculate_snr(multi_conv_data_left_no_noise[0,...],y_dsbf[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_dsbf[1,...])
snr_doa_mvdr_post=calculate_snr(multi_conv_data_left_no_noise[0,...],y_mvdr[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_mvdr[1,...])
snr_doa_mvdr2_post=calculate_snr(multi_conv_data_left_no_noise[0,...],y_mvdr2[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_mvdr2[1,...])
snr_doa_max_snr_post=calculate_snr(multi_conv_data_left_no_noise[0,...],y_max_snr[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_max_snr[1,...])
snr_doa_mwf_post=calculate_snr(multi_conv_data_left_no_noise[0,...],y_mwf[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_mwf[1,...])

snr_pre/=2.
snr_doa_sparse_post/=2.
snr_doa_dsbf_post/=2.
snr_doa_mvdr_post/=2.
snr_doa_mvdr2_post/=2.
snr_doa_max_snr_post/=2.
snr_doa_mwf_post/=2.

#ファイルに書き込む
write_file_from_time_signal(y_sparse[0,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_sparse_left.wav",sample_rate)
write_file_from_time_signal(y_sparse[1,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_sparse_right.wav",sample_rate)
write_file_from_time_signal(y_dsbf[0,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_dsbf_left.wav",sample_rate)
write_file_from_time_signal(y_dsbf[1,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_dsbf_right.wav",sample_rate)
write_file_from_time_signal(y_mvdr[0,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_mvdr_left.wav",sample_rate)
write_file_from_time_signal(y_mvdr[1,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_mvdr_right.wav",sample_rate)
write_file_from_time_signal(y_mvdr2[0,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_mvdr2_left.wav",sample_rate)
write_file_from_time_signal(y_mvdr2[1,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_mvdr2_right.wav",sample_rate)
write_file_from_time_signal(y_max_snr[0,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_max_snr_left.wav",sample_rate)
write_file_from_time_signal(y_max_snr[1,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_max_snr_right.wav",sample_rate)
write_file_from_time_signal(y_mwf[0,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_mwf_left.wav",sample_rate)
write_file_from_time_signal(y_mwf[1,...]*np.iinfo(np.int16).max/20.,"./sparse_doa_mwf_right.wav",sample_rate)



print("method:    ", "DOA_SPARSE","DOA_DSBF","DOA_MVDR", "DOA_MVDR2", "DOA_MAX_SNR","DOA_MWF")

print("Δsnr [dB]: {:.2f}       {:.2f}     {:.2f}    {:.2f}     {:.2f}       {:.2f}".format(snr_doa_sparse_post-snr_pre,snr_doa_dsbf_post-snr_pre,snr_doa_mvdr_post-snr_pre,snr_doa_mvdr2_post-snr_pre,snr_doa_max_snr_post-snr_pre,snr_doa_mwf_post-snr_pre))

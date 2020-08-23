

import wave as wave
import pyroomacoustics as pa
import numpy as np
import scipy.signal as sp
import scipy as scipy

#順列計算に使用
import itertools
import time


#コントラスト関数の微分（球対称多次元ラプラス分布を仮定）
#s_hat: 分離信号(M, Nk, Lt)
def phi_multivariate_laplacian(s_hat):

    power=np.square(np.abs(s_hat))
    norm=np.sqrt(np.sum(power,axis=1,keepdims=True))

    phi=s_hat/np.maximum(norm,1.e-18)
    return(phi)

#コントラスト関数の微分（球対称ラプラス分布を仮定）
#s_hat: 分離信号(M, Nk, Lt)
def phi_laplacian(s_hat):

    norm=np.abs(s_hat)
    phi=s_hat/np.maximum(norm,1.e-18)
    return(phi)

#コントラスト関数（球対称ラプラス分布を仮定）
#s_hat: 分離信号(M, Nk, Lt)
def contrast_laplacian(s_hat):

    norm=2.*np.abs(s_hat)
    
    return(norm)

#コントラスト関数（球対称多次元ラプラス分布を仮定）
#s_hat: 分離信号(M, Nk, Lt)
def contrast_multivariate_laplacian(s_hat):
    power=np.square(np.abs(s_hat))
    norm=2.*np.sqrt(np.sum(power,axis=1,keepdims=True))
  
    return(norm)

#ICAによる分離フィルタ更新
#x:入力信号( M, Nk, Lt)
#W: 分離フィルタ(Nk,M,M)
#mu: 更新係数
#n_ica_iterations: 繰り返しステップ数
#phi_func: コントラスト関数の微分を与える関数
#contrast_func: コントラスト関数
#is_use_non_holonomic: True (非ホロノミック拘束を用いる） False (用いない）
#return W 分離フィルタ(Nk,M,M) s_hat 出力信号(M,Nk, Lt),cost_buff ICAのコスト (T)
def execute_natural_gradient_ica(x,W,phi_func=phi_laplacian,contrast_func=contrast_laplacian,mu=1.0,n_ica_iterations=20,is_use_non_holonomic=True):
    
    #マイクロホン数を取得する
    M=np.shape(x)[0]

    cost_buff=[]
    for t in range(n_ica_iterations):
        #音源分離信号を得る
        s_hat=np.einsum('kmn,nkt->mkt',W,x)

        #コントラスト関数を計算
        G=contrast_func(s_hat)

        #コスト計算
        cost=np.sum(np.mean(G,axis=-1))-np.sum(2.*np.log(np.abs(np.linalg.det(W)) ))
        cost_buff.append(cost)

        #コンストラクト関数の微分を取得
        phi=phi_func(s_hat)

        phi_s=np.einsum('mkt,nkt->ktmn',phi,np.conjugate(s_hat))
        phi_s=np.mean(phi_s,axis=1)

        I=np.eye(M,M)
        if is_use_non_holonomic==False:
            deltaW=np.einsum('kmi,kin->kmn',I[None,...]-phi_s,W)
        else:
            mask=(np.ones((M,M))-I)[None,...]
            deltaW=np.einsum('kmi,kin->kmn',np.multiply(mask,-phi_s),W)
            
        #フィルタを更新する
        W=W+mu*deltaW
    
    #最後に出力信号を分離
    s_hat=np.einsum('kmn,nkt->mkt',W,x)

    return(W,s_hat,cost_buff)

#IP法による分離フィルタ更新
#x:入力信号( M, Nk, Lt)
#W: 分離フィルタ(Nk,M,M)
#n_iterations: 繰り返しステップ数
#return W 分離フィルタ(Nk,M,M) s_hat 出力信号(M,Nk, Lt),cost_buff コスト (T)
def execute_ip_multivariate_laplacian_iva(x,W,n_iterations=20):
    
    #マイクロホン数を取得する
    M=np.shape(x)[0]

    cost_buff=[]
    for t in range(n_iterations):
        
        #音源分離信号を得る
        s_hat=np.einsum('kmn,nkt->mkt',W,x)

        #補助変数を更新する
        v=np.sqrt(np.sum(np.square(np.abs(s_hat)),axis=1))

        #コントラスト関数を計算
        G=contrast_multivariate_laplacian(s_hat)

        #コスト計算
        cost=np.sum(np.mean(G,axis=-1))-np.sum(2.*np.log(np.abs(np.linalg.det(W)) ))
        cost_buff.append(cost)

        #IP法による更新
        Q=np.einsum('st,mkt,nkt->tksmn',1./np.maximum(v,1.e-18),x,np.conjugate(x))
        Q=np.average(Q,axis=0)
        
        for source_index in range(M):
            WQ=np.einsum('kmi,kin->kmn',W,Q[:,source_index,:,:])
            invWQ=np.linalg.pinv(WQ)
            W[:,source_index,:]=np.conjugate(invWQ[:,:,source_index])
            wVw=np.einsum('km,kmn,kn->k',W[:,source_index,:],Q[:,source_index,:,:],np.conjugate(W[:,source_index,:]))
            wVw=np.sqrt(np.abs(wVw))
            W[:,source_index,:]=W[:,source_index,:]/np.maximum(wVw[:,None],1.e-18)

 
    s_hat=np.einsum('kmn,nkt->mkt',W,x)

    return(W,s_hat,cost_buff)

#周波数間の振幅相関に基づくパーミュテーション解法
#s_hat: M,Nk,Lt
#return permutation_index_result：周波数毎のパーミュテーション解 
def solver_inter_frequency_permutation(s_hat):
    n_sources=np.shape(s_hat)[0]
    n_freqs=np.shape(s_hat)[1]
    n_frames=np.shape(s_hat)[2]

    s_hat_abs=np.abs(s_hat)

    norm_amp=np.sqrt(np.sum(np.square(s_hat_abs),axis=0,keepdims=True))
    s_hat_abs=s_hat_abs/np.maximum(norm_amp,1.e-18)

    spectral_similarity=np.einsum('mkt,nkt->k',s_hat_abs,s_hat_abs)
    
    frequency_order=np.argsort(spectral_similarity)
    
    #音源間の相関が最も低い周波数からパーミュテーションを解く
    is_first=True
    permutations=list(itertools.permutations(range(n_sources)))
    permutation_index_result={}
    
    for freq in frequency_order:
        
        if is_first==True:
            is_first=False

            #初期値を設定する
            accumurate_s_abs=s_hat_abs[:,frequency_order[0],:]
            permutation_index_result[freq]=range(n_sources)
        else:
            max_correlation=0
            max_correlation_perm=None
            for perm in permutations:
                s_hat_abs_temp=s_hat_abs[list(perm),freq,:]
                correlation=np.sum(accumurate_s_abs*s_hat_abs_temp)
                
                
                if max_correlation_perm is None:
                    max_correlation_perm=list(perm)
                    max_correlation=correlation
                elif max_correlation < correlation:
                    max_correlation=correlation
                    max_correlation_perm=list(perm)
            permutation_index_result[freq]=max_correlation_perm
            accumurate_s_abs+=s_hat_abs[max_correlation_perm,freq,:]
   
    return(permutation_index_result)
    
#プロジェクションバックで最終的な出力信号を求める
#s_hat: M,Nk,Lt
#W: 分離フィルタ(Nk,M,M)
#retunr c_hat: マイクロホン位置での分離結果(M,M,Nk,Lt)
def projection_back(s_hat,W):
    
    #ステアリングベクトルを推定
    A=np.linalg.pinv(W)
    c_hat=np.einsum('kmi,ikt->mikt',A,s_hat)
    return(c_hat)


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

is_use_reverb=False

if is_use_reverb==False:
    # 部屋を生成する
    room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0) 
    room_no_noise_left = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)
    room_no_noise_right = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)

else:

    room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=17,absorption=0.4)
    room_no_noise_left = pa.ShoeBox(room_dim, fs=sample_rate, max_order=17,absorption=0.4)
    room_no_noise_right = pa.ShoeBox(room_dim, fs=sample_rate, max_order=17,absorption=0.4)

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
write_file_from_time_signal(multi_conv_data_left_no_noise[0,:]*np.iinfo(np.int16).max/20.,"./ica_left_clean.wav",sample_rate)

#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data_right_no_noise[0,:]*np.iinfo(np.int16).max/20.,"./ica_right_clean.wav",sample_rate)

#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data[0,:]*np.iinfo(np.int16).max/20.,"./ica_in_left.wav",sample_rate)
write_file_from_time_signal(multi_conv_data[0,:]*np.iinfo(np.int16).max/20.,"./ica_in_right.wav",sample_rate)

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(multi_conv_data,fs=sample_rate,window="hann",nperseg=N)

#ICAの繰り返し回数
n_ica_iterations=50

#ICAの分離フィルタを初期化
Wica=np.zeros(shape=(Nk,n_sources,n_sources),dtype=np.complex)

Wica=Wica+np.eye(n_sources)[None,...]

Wiva=Wica.copy()
Wiva_ip=Wica.copy()

start_time=time.time()
#自然勾配法に基づくIVA実行コード（引数に与える関数を変更するだけ)
Wiva,s_iva,cost_buff_iva=execute_natural_gradient_ica(stft_data,Wiva,phi_func=phi_multivariate_laplacian,contrast_func=contrast_multivariate_laplacian,mu=0.1,n_ica_iterations=n_ica_iterations,is_use_non_holonomic=False)
y_iva=projection_back(s_iva,Wiva)
iva_time=time.time()

#IP法に基づくIVA実行コード（引数に与える関数を変更するだけ)
Wiva_ip,s_iva_ip,cost_buff_iva_ip=execute_ip_multivariate_laplacian_iva(stft_data,Wiva_ip,n_iterations=n_ica_iterations)
y_iva_ip=projection_back(s_iva_ip,Wiva_ip)
iva_ip_time=time.time()

Wica,s_ica,cost_buff_ica=execute_natural_gradient_ica(stft_data,Wica,mu=0.1,n_ica_iterations=n_ica_iterations,is_use_non_holonomic=False)
permutation_index_result=solver_inter_frequency_permutation(s_ica)
y_ica=projection_back(s_ica,Wica)

#パーミュテーションを解く
for k in range(Nk):
    y_ica[:,:,k,:]=y_ica[:,permutation_index_result[k],k,:]

ica_time=time.time()

t,y_ica=sp.istft(y_ica[0,...],fs=sample_rate,window="hann",nperseg=N)
t,y_iva=sp.istft(y_iva[0,...],fs=sample_rate,window="hann",nperseg=N)
t,y_iva_ip=sp.istft(y_iva_ip[0,...],fs=sample_rate,window="hann",nperseg=N)

snr_pre=calculate_snr(multi_conv_data_left_no_noise[0,...],multi_conv_data[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],multi_conv_data[0,...])
snr_pre/=2.

snr_ica_post1=calculate_snr(multi_conv_data_left_no_noise[0,...],y_ica[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_ica[1,...])
snr_ica_post2=calculate_snr(multi_conv_data_left_no_noise[0,...],y_ica[1,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_ica[0,...])

snr_ica_post=np.maximum(snr_ica_post1,snr_ica_post2)
snr_ica_post/=2.

snr_iva_post1=calculate_snr(multi_conv_data_left_no_noise[0,...],y_iva[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_iva[1,...])
snr_iva_post2=calculate_snr(multi_conv_data_left_no_noise[0,...],y_iva[1,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_iva[0,...])

snr_iva_post=np.maximum(snr_iva_post1,snr_iva_post2)
snr_iva_post/=2.

snr_iva_ip_post1=calculate_snr(multi_conv_data_left_no_noise[0,...],y_iva_ip[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_iva_ip[1,...])
snr_iva_ip_post2=calculate_snr(multi_conv_data_left_no_noise[0,...],y_iva_ip[1,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_iva_ip[0,...])

snr_iva_ip_post=np.maximum(snr_iva_ip_post1,snr_iva_ip_post2)
snr_iva_ip_post/=2.

write_file_from_time_signal(y_ica[0,...]*np.iinfo(np.int16).max/20.,"./ica_1.wav",sample_rate)
write_file_from_time_signal(y_ica[1,...]*np.iinfo(np.int16).max/20.,"./ica_2.wav",sample_rate)

write_file_from_time_signal(y_iva[0,...]*np.iinfo(np.int16).max/20.,"./iva_1.wav",sample_rate)
write_file_from_time_signal(y_iva[1,...]*np.iinfo(np.int16).max/20.,"./iva_2.wav",sample_rate)

write_file_from_time_signal(y_iva_ip[0,...]*np.iinfo(np.int16).max/20.,"./iva_ip_1.wav",sample_rate)
write_file_from_time_signal(y_iva_ip[1,...]*np.iinfo(np.int16).max/20.,"./iva_ip_2.wav",sample_rate)


print("method:    ", "NG-ICA", "NG-IVA", "AuxIVA")
print("処理時間[sec]: {:.2f}  {:.2f}  {:.2f}".format(ica_time-iva_ip_time,iva_ip_time-iva_time,iva_time-start_time))
print("Δsnr [dB]: {:.2f}  {:.2f}  {:.2f}".format(snr_ica_post-snr_pre,snr_iva_post-snr_pre,snr_iva_ip_post-snr_pre))

#コストの値を表示
#for t in range(n_ica_iterations):
#    print(t,cost_buff_ica[t],cost_buff_iva[t],cost_buff_iva_ip[t])



import wave as wave
import pyroomacoustics as pa
import numpy as np
import scipy.signal as sp
import scipy as scipy

#順列計算に使用
import itertools
import time

#A: ...mn
#B: ...ij
#AとBの最後の二軸以外の次元は一致していることを前提とする
def batch_kron(A,B):
 if np.shape(A)[:-2]!=np.shape(B)[:-2]:
     print("error")
     return None
 else:
    return(np.reshape(np.einsum("...mn,...ij->...minj",A,B),np.shape(A)[:-2]+(np.shape(A)[-2]*np.shape(B)[-2],np.shape(A)[-1]*np.shape(B)[-1])))


#x:入力信号( M, Nk, Lt)
#D:遅延フレーム数
#Lh:残響除去フィルタのタップ長
#return x_bar: 過去のマイク入力信号(Lh,M,Nk,Lt)
def make_x_bar(x,D,Lh):
    
    #フレーム数を取得
    Lt=np.shape(x)[2]

    #過去のマイク入力信号の配列を準備
    x_bar=np.zeros(shape=(Lh,)+np.shape(x),dtype=np.complex)

    for tau in range(Lh):
        x_bar[tau,...,tau+D:]=x[:,:,:-(tau+D)]

    return(x_bar)

#IP法によるLGMのパラメータ推定法
#x:入力信号( M, Nk, Lt)
#Ns: 音源数
#n_iterations: 繰り返しステップ数
#return R 共分散行列(Nk,Ns,M,M) v 時間周波数分散(Nk,Ns,Lt),c_bar 音源分離信号(M,Ns,Nk,Lt), cost_buff コスト (T)
def execute_mm_lgm(x,Ns=2,n_iterations=20):
    
    #マイクロホン数・周波数・フレーム数を取得する
    M=np.shape(x)[0]
    Nk=np.shape(x)[1]
    Lt=np.shape(x)[2]

    #Rとvを初期化する
    mask=np.random.uniform(size=Nk*Ns*Lt)
    mask=np.reshape(mask,(Nk,Ns,Lt))
    R=np.einsum("kst,mkt,nkt->kstmn",mask,x,np.conjugate(x))
    R=np.average(R,axis=2)
    v=np.random.uniform(size=Nk*Ns*Lt)
    v=np.reshape(v,(Nk,Ns,Lt))

    cost_buff=[]
    for t in range(n_iterations):
        
        #入力信号の共分散行列を求める
        vR=np.einsum("kst,ksmn->kstmn",v,R)
        V=np.sum(vR,axis=1)
        V_inverse=np.linalg.pinv(V)

        #コスト計算
        cost=np.sum(np.einsum("mkt,ktmn,nkt->kt",np.conjugate(x),V_inverse,x) +np.log(np.abs(np.linalg.det(V))))
        cost/=np.float(Lt)
        cost=np.real(cost)
        cost_buff.append(cost)

        #パラメータを更新
        
        #Rの更新
        V_inverseX=np.einsum('ktmn,nkt->ktm',V_inverse,x)
        V_inverseXV_inverseX=np.einsum('ktm,ktn->ktmn',V_inverseX,np.conjugate(V_inverseX))
        A=np.einsum('kst,ktmn->ksmn',v,V_inverse)
        B=np.einsum('kst,ktmn->ksmn',v,V_inverseXV_inverseX)
        RBR=np.einsum('ksmn,ksni,ksij->ksmj',R,B,R)
        invA=np.linalg.pinv(A)
        A_RBR=np.matmul(A,RBR)
        R=np.concatenate([np.concatenate([np.matmul(invA[k,s,...],scipy.linalg.sqrtm(A_RBR[k,s,...]))[None,None,...] for k in range(Nk)],axis=0) for s in range(Ns)],axis=1)
        R=(R+np.transpose(np.conjugate(R),[0,1,3,2]))/(2.0+0.0j)

        #vの更新
        v=v*np.sqrt(np.einsum('ktm,ktn,ksnm->kst',V_inverseX,np.conjugate(V_inverseX),R)/np.maximum(np.einsum('ktmn,ksnm->kst',V_inverse,R) ,1.e-18))

       
    vR=np.einsum("kst,ksmn->kstmn",v,R)
    V=np.sum(vR,axis=1)
    V_inverse=np.linalg.pinv(V)
    Wmwf=np.einsum("kstmi,ktin->kstmn",vR,V_inverse)

    #音源分離信号を得る
    c_bar=np.einsum('kstmn,nkt->mskt',Wmwf,x)

    return(R,v,c_bar,cost_buff)

#LGMの音源分離と残響除去のパラメータ推定法
#x:入力信号( M, Nk, Lt)
#x_bar:過去のマイク入力信号(Lh,M, Nk, Lt)
#Ns: 音源数
#n_iterations: 繰り返しステップ数
#return R 共分散行列(Nk,Ns,M,M) v 時間周波数分散(Nk,Ns,Lt),c_bar 音源分離信号(M,Ns,Nk,Lt), cost_buff コスト (T)
def execute_mm_lgm_dereverb(x,x_bar,Ns=2,n_iterations=20):
    
    #マイクロホン数・周波数・フレーム数を取得する
    M=np.shape(x)[0]
    Nk=np.shape(x)[1]
    Lt=np.shape(x)[2]
    Lh=np.shape(x_bar)[0]

    x_bar=np.reshape(x_bar,[Lh*M,Nk,Lt])

    #Rとvを初期化する
    mask=np.random.uniform(size=Nk*Ns*Lt)
    mask=np.reshape(mask,(Nk,Ns,Lt))
    R=np.einsum("kst,mkt,nkt->kstmn",mask,x,np.conjugate(x))
    R=np.average(R,axis=2)
    v=np.random.uniform(size=Nk*Ns*Lt)
    v=np.reshape(v,(Nk,Ns,Lt))

    #共分散行列を計算
    x_bar_x_bar_H=np.einsum('ikt,jkt->ktij',x_bar,np.conjugate(x_bar))

    #相関行列を計算
    x_bar_x_H=np.einsum('ikt,mkt->ktim',x_bar,np.conjugate(x))

    cost_buff=[]
    for t in range(n_iterations):
        #入力信号の共分散行列を求める
        vR=np.einsum("kst,ksmn->kstmn",v,R)
        
        V=np.sum(vR,axis=1)
        V=V+np.eye(M,M)*1.e-8
        V_inverse=np.linalg.inv(V)
        
        #残響除去フィルタを求める
        x_barx_H_V_inv=np.einsum("ktim,ktmn->kin",x_bar_x_H,V_inverse)

        vec_x_bar_x_HV_inv=np.reshape(np.transpose(x_barx_H_V_inv,[0,2,1]),(Nk,Lh*M*M))
        
        #多次元配列対応版のクロネッカー積
        V_inverse_x_x_H=batch_kron(np.transpose(V_inverse,(0,1,3,2)),x_bar_x_bar_H)

        #vecHを求める
        vec_h=np.einsum("kmr,kr->km",np.linalg.inv(np.sum(V_inverse_x_x_H,axis=1)), vec_x_bar_x_HV_inv)
        
        #行列に戻す
        h=np.transpose(np.reshape(vec_h,(Nk,M,Lh*M)),(0,2,1))

        #残響除去を行う
        x_reverb=np.einsum('kjm,jkt->mkt',np.conjugate(h),x_bar)
        x_dereverb=x-x_reverb
        

        #コスト計算
        cost=np.sum(np.einsum("mkt,ktmn,nkt->kt",np.conjugate(x_dereverb),V_inverse,x_dereverb) +np.log(np.abs(np.linalg.det(V))))
        cost/=np.float(Lt)
        cost=np.real(cost)
        cost_buff.append(cost)
        #print(t,cost)
        #パラメータを更新
        
        #Rの更新
        V_inverseX=np.einsum('ktmn,nkt->ktm',V_inverse,x_dereverb)
        V_inverseXV_inverseX=np.einsum('ktm,ktn->ktmn',V_inverseX,np.conjugate(V_inverseX))
        A=np.einsum('kst,ktmn->ksmn',v,V_inverse)
        B=np.einsum('kst,ktmn->ksmn',v,V_inverseXV_inverseX)
        RBR=np.einsum('ksmn,ksni,ksij->ksmj',R,B,R)
        invA=np.linalg.pinv(A)
        A_RBR=np.matmul(A,RBR)
        R=np.concatenate([np.concatenate([np.matmul(invA[k,s,...],scipy.linalg.sqrtm(A_RBR[k,s,...]))[None,None,...] for k in range(Nk)],axis=0) for s in range(Ns)],axis=1)
        R=(R+np.transpose(np.conjugate(R),[0,1,3,2]))/(2.0+0.0j)

        #vの更新
        v=v*np.sqrt(np.einsum('ktm,ktn,ksnm->kst',V_inverseX,np.conjugate(V_inverseX),R)/np.maximum(np.einsum('ktmn,ksnm->kst',V_inverse,R) ,1.e-18))

       
    vR=np.einsum("kst,ksmn->kstmn",v,R)
    V=np.sum(vR,axis=1)
    V_inverse=np.linalg.pinv(V)
    Wmwf=np.einsum("kstmi,ktin->kstmn",vR,V_inverse)

    #音源分離信号を得る
    c_bar=np.einsum('kstmn,nkt->mskt',Wmwf,x_dereverb)

    return(R,v,c_bar,cost_buff)

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

#フレームシフト
Nshift=int(N/4)

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

is_use_reverb=True

if is_use_reverb==False:
    # 部屋を生成する
    room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0) 
    room_no_noise_left = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)
    room_no_noise_right = pa.ShoeBox(room_dim, fs=sample_rate, max_order=0)

else:

    room = pa.ShoeBox(room_dim, fs=sample_rate, max_order=17,absorption=0.4)
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
write_file_from_time_signal(multi_conv_data_left_no_noise[0,:]*np.iinfo(np.int16).max/20.,"./lgm_dereverb_left_clean.wav",sample_rate)

#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data_right_no_noise[0,:]*np.iinfo(np.int16).max/20.,"./lgm_dereverb_right_clean.wav",sample_rate)

#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data[0,:]*np.iinfo(np.int16).max/20.,"./lgm_dereverb_in_left.wav",sample_rate)
write_file_from_time_signal(multi_conv_data[0,:]*np.iinfo(np.int16).max/20.,"./lgm_dereverb_in_right.wav",sample_rate)

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(multi_conv_data,fs=sample_rate,window="hann",nperseg=N,noverlap=N-Nshift)

#ICAの繰り返し回数
n_ica_iterations=50

#残響除去のパラメータ
D=2
Lh=5

#過去のマイクロホン入力信号
x_bar=make_x_bar(stft_data,D,Lh)

#処理するフレーム数
Lt=np.shape(stft_data)[-1]

#MM法に基づくLGM+Dereverb実行コード
Rlgm_mm_dereverb,vlgm_mm_dereverb,y_lgm_mm_dereverb,cost_buff_lgm_mm_dereverb=execute_mm_lgm_dereverb(stft_data,x_bar,Ns=n_sources,n_iterations=n_ica_iterations)
permutation_index_result=solver_inter_frequency_permutation(y_lgm_mm_dereverb[0,...])

#パーミュテーションを解く
for k in range(Nk):
    y_lgm_mm_dereverb[:,:,k,:]=y_lgm_mm_dereverb[:,permutation_index_result[k],k,:]

#MM法に基づくLGM実行コード
Rlgm_mm,vlgm_mm,y_lgm_mm,cost_buff_lgm_mm=execute_mm_lgm(stft_data,Ns=n_sources,n_iterations=n_ica_iterations)
permutation_index_result=solver_inter_frequency_permutation(y_lgm_mm[0,...])
for k in range(Nk):
    y_lgm_mm[:,:,k,:]=y_lgm_mm[:,permutation_index_result[k],k,:]

t,y_lgm_mm=sp.istft(y_lgm_mm[0,...],fs=sample_rate,window="hann",nperseg=N,noverlap=N-Nshift)
t,y_lgm_mm_dereverb=sp.istft(y_lgm_mm_dereverb[0,...],fs=sample_rate,window="hann",nperseg=N,noverlap=N-Nshift)

snr_pre=calculate_snr(multi_conv_data_left_no_noise[0,...],multi_conv_data[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],multi_conv_data[0,...])
snr_pre/=2.

snr_lgm_mm_post1=calculate_snr(multi_conv_data_left_no_noise[0,...],y_lgm_mm[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_lgm_mm[1,...])
snr_lgm_mm_post2=calculate_snr(multi_conv_data_left_no_noise[0,...],y_lgm_mm[1,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_lgm_mm[0,...])

snr_lgm_mm_post=np.maximum(snr_lgm_mm_post1,snr_lgm_mm_post2)
snr_lgm_mm_post/=2.


snr_lgm_mm_dereverb_post1=calculate_snr(multi_conv_data_left_no_noise[0,...],y_lgm_mm_dereverb[0,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_lgm_mm_dereverb[1,...])
snr_lgm_mm_dereverb_post2=calculate_snr(multi_conv_data_left_no_noise[0,...],y_lgm_mm_dereverb[1,...])+calculate_snr(multi_conv_data_right_no_noise[0,...],y_lgm_mm_dereverb[0,...])

snr_lgm_mm_dereverb_post=np.maximum(snr_lgm_mm_dereverb_post1,snr_lgm_mm_dereverb_post2)
snr_lgm_mm_dereverb_post/=2.

write_file_from_time_signal(y_lgm_mm[0,...]*np.iinfo(np.int16).max/20.,"./lgm_mm_1.wav",sample_rate)
write_file_from_time_signal(y_lgm_mm[1,...]*np.iinfo(np.int16).max/20.,"./lgm_mm_2.wav",sample_rate)

write_file_from_time_signal(y_lgm_mm_dereverb[0,...]*np.iinfo(np.int16).max/20.,"./lgm_mm_dereverb_1.wav",sample_rate)
write_file_from_time_signal(y_lgm_mm_dereverb[1,...]*np.iinfo(np.int16).max/20.,"./lgm_mm_dereverb_2.wav",sample_rate)


print("method:    ", "LGM-MM","LGM-Dereverb-MM")
print("Δsnr [dB]:  {:.2f}  {:.2f}".format(snr_lgm_mm_post-snr_pre,snr_lgm_mm_dereverb_post-snr_pre))

#コストの値を表示
#for t in range(n_ica_iterations):
#    print(t,cost_buff_lgm_mm[t],cost_buff_lgm_mm_dereverb[t])



import numpy as np

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

#サンプリングレート [Hz]
sample_rate=16000

#フレームサイズ
N=1024

#周波数の数
Nk=N/2+1

#各ビンの周波数
freqs=np.arange(0,Nk,1)*sample_rate/N

#マイクロホンアレイのマイク配置
mic_alignments = np.array(
    [
        [-0.01, 0.0, 0.0],
        [0.01, 0.0, 0.0],
    ]
).T

#音源の方向
doas=np.array(
    [[np.pi/2,0],
     [np.pi/2,np.pi]
    ]    )

#音源とマイクロホンの距離
distance=1

#音源の位置ベクトル
source_locations=np.zeros((3, doas.shape[0]), dtype=doas.dtype)
source_locations[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[2, :] = np.cos(doas[:, 0])
source_locations *= distance

#Near仮定に基づくステアリングベクトルを計算: steering_vectors(Nk x Ns x M)
near_steering_vectors=calculate_steering_vector(mic_alignments,source_locations,freqs,is_use_far=False)

#Far仮定に基づくステアリングベクトルを計算: steering_vectors(Nk x Ns x M)
far_steering_vectors=calculate_steering_vector(mic_alignments,source_locations,freqs,is_use_far=True)

#内積を計算
inner_product=np.einsum("ksm,ksm->ks",np.conjugate(near_steering_vectors),far_steering_vectors)
print(np.average(np.abs(inner_product)))

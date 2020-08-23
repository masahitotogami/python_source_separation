

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

#遅延和アレイを実行する
#x_left:左の音源に近いマイクロホン(Nk, Lt)
#x_right:右の音源に近いマイクロホン(Nk,Lt)
#is_use_amplitude: 振幅差を使って分離を行う場合はTrue
#return y_left 左のマイクに近い音源の出力信号(Nk, Lt)
#       y_right 右のマイクに近い音源の出力信号(Nk,Lt)
def execute_two_microphone_sparse_separation(x_left,x_right,is_use_amplitude=False):

    if is_use_amplitude==True:
        #振幅比率を使った分離
        amp_ratio=np.abs(x_left)/np.maximum(np.abs(x_right),1.e-18)

        y_left=(amp_ratio > 1.).astype(np.float)*x_left

        y_right=(amp_ratio <= 1.).astype(np.float)*x_right

    else:
        #位相差を用いた分離
        phase_difference=np.angle(x_left/x_right)
        
        y_left=(phase_difference > 0.).astype(np.float)*x_left
        
        y_right=(phase_difference <= 0.).astype(np.float)*x_right

    return(y_left,y_right)


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
mic_alignments = np.array(
    [
        [x, 0.0, 0.0] for x in np.arange(-0.2,0.21,0.4)
    ]
)



#マイクロホン数
n_channels=np.shape(mic_alignments)[0]

#マイクロホンアレイの座標
R=mic_alignments .T+mic_array_loc[:,None]

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
     [np.pi/2., 0.]
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
write_file_from_time_signal(multi_conv_data_left_no_noise[0,:]*np.iinfo(np.int16).max/20.,"./near_left_clean.wav",sample_rate)

#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data_right_no_noise[1,:]*np.iinfo(np.int16).max/20.,"./near_right_clean.wav",sample_rate)

#畳み込んだ波形をファイルに書き込む
write_file_from_time_signal(multi_conv_data[0,:]*np.iinfo(np.int16).max/20.,"./near_in_left.wav",sample_rate)
write_file_from_time_signal(multi_conv_data[1,:]*np.iinfo(np.int16).max/20.,"./near_in_right.wav",sample_rate)

#短時間フーリエ変換を行う
f,t,stft_data=sp.stft(multi_conv_data,fs=sample_rate,window="hann",nperseg=N)

#位相差もしくは振幅差で分離
y_phase_left,y_phase_right=execute_two_microphone_sparse_separation(stft_data[0,...],stft_data[1,...],False)
y_amp_left,y_amp_right=execute_two_microphone_sparse_separation(stft_data[0,...],stft_data[1,...],True)

#時間領域の波形に戻す
t,y_phase_left=sp.istft(y_phase_left,fs=sample_rate,window="hann",nperseg=N)
t,y_phase_right=sp.istft(y_phase_right,fs=sample_rate,window="hann",nperseg=N)
t,y_amp_left=sp.istft(y_amp_left,fs=sample_rate,window="hann",nperseg=N)
t,y_amp_right=sp.istft(y_amp_right,fs=sample_rate,window="hann",nperseg=N)


#SNRをはかる
snr_pre=calculate_snr(multi_conv_data_left_no_noise[0,...],multi_conv_data[0,...])+calculate_snr(multi_conv_data_right_no_noise[1,...],multi_conv_data[1,...])
snr_phase_post=calculate_snr(multi_conv_data_left_no_noise[0,...],y_phase_left)+calculate_snr(multi_conv_data_right_no_noise[1,...],y_phase_right)
snr_amp_post=calculate_snr(multi_conv_data_left_no_noise[0,...],y_amp_left)+calculate_snr(multi_conv_data_right_no_noise[1,...],y_amp_right)
snr_pre/=2.
snr_phase_post/=2.
snr_amp_post/=2.


#ファイルに書き込む
write_file_from_time_signal(y_phase_left*np.iinfo(np.int16).max/20.,"./near_sparse_phase_left.wav",sample_rate)
write_file_from_time_signal(y_phase_right*np.iinfo(np.int16).max/20.,"./near_sparse_phase_right.wav",sample_rate)
write_file_from_time_signal(y_amp_left*np.iinfo(np.int16).max/20.,"./near_sparse_amp_left.wav",sample_rate)
write_file_from_time_signal(y_amp_right*np.iinfo(np.int16).max/20.,"./near_sparse_amp_right.wav",sample_rate)


print("method:    ", "PHASE", "AMPLITUDE")

print("Δsnr [dB]: {:.2f} {:.2f} ".format(snr_phase_post-snr_pre,snr_amp_post-snr_pre))

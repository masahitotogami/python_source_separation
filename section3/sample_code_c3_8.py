
#numpyをインポート（行列を扱う各種関数を含む）
import numpy as np

#乱数の種を設定
np.random.seed(0)

#行列の大きさを設定する
L=10
M=3
N=3
#ランダムな複素数のテンソル(ndarray)を定義する
A=np.random.uniform(size=L*M*N)+np.random.uniform(size=L*M*N)*1.j
A=np.reshape(A,(L,M,N))

#正定エルミート行列を作る
B=np.einsum("lmk,lnk->lmn",A,np.conjugate(A))

#Aの固有値分解実施
w,v=np.linalg.eig(A)

#固有値と固有ベクトルからAを復元できるか検証
A_reconst=np.einsum("lmk,lk,lkn->lmn",v,w,np.linalg.inv(v))
print("A[0]: \n",A[0])
print("A_reconst[0]: \n",A_reconst[0])

#Bの固有値分解実施
w,v=np.linalg.eigh(B)

#固有値と固有ベクトルからBを復元できるか検証
B_reconst=np.einsum("lmk,lk,lnk->lmn",v,w,np.conjugate(v))
print("B[0]: \n",B[0])
print("B[0]: \n",B_reconst[0])
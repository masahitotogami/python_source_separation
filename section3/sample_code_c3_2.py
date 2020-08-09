

#numpyをインポート（行列を扱う各種関数を含む）
import numpy as np

#乱数の種を設定
np.random.seed(0)

#テンソルの大きさを設定
L=10
K=5
M=3
R=3
N=3
#ランダムな複素数のテンソル(ndarray)を定義する
A=np.random.uniform(size=L*K*M*R)+np.random.uniform(size=L*K*M*R)*1.j
A=np.reshape(A,(L,K,M,R))

B=np.random.uniform(size=K*R*N)+np.random.uniform(size=K*R*N)*1.j
B=np.reshape(B,(K,R,N))

#einsumを使って行列積を計算する
C=np.einsum("lkmr,krn->lkmn",A,B)

#行列Cの大きさを表示
print("shape(C): ",np.shape(C))

#l=0,k=0の要素について検算実施
print("A(0,0)B(0,0)=\n",np.matmul(A[0,0,...],B[0,...]))
print("C(0,0)=\n",C[0,0,...])


#einsumを使って行列積をl,k毎に計算した後、かつl方向に和を取る
C=np.einsum("lkmr,krn->kmn",A,B)

#行列Cの大きさを表示
print("shape(C): ",np.shape(C))

#k=0の要素について検算実施
for l in range(L):
    if l==0:
        C_2=np.matmul(A[l,0,...],B[0,...])
    else:
        C_2=C_2+np.matmul(A[l,0,...],B[0,...])
        
print("C_2(0)=\n",C_2)
print("C(0)=\n",C[0,...])

#einsumを使ってアダマール積を計算する
C=np.einsum("lkmn,kmn->lkmn",A,B)

#行列Cの大きさを表示
print("shape(C): ",np.shape(C))

#l=0,k=0の要素について検算実施
print("A(0,0)B(0,0)=\n",np.multiply(A[0,0,...],B[0,...]))
print("C(0,0)=\n",C[0,0,...])



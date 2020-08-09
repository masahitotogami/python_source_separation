

#numpyをインポート（行列を扱う各種関数を含む）
import numpy as np

#乱数の種を設定
np.random.seed(0)

#テンソルの大きさを設定
L=10
K=5
M=3
R=3
S=3
N=3
#ランダムな複素数のテンソル(ndarray)を定義する
A=np.random.uniform(size=L*K*M*R)+np.random.uniform(size=L*K*M*R)*1.j
A=np.reshape(A,(L,K,M,R))

B=np.random.uniform(size=K*R*S)+np.random.uniform(size=K*R*S)*1.j
B=np.reshape(B,(K,R,S))


C=np.random.uniform(size=L*S*N)+np.random.uniform(size=L*S*N)*1.j
C=np.reshape(C,(L,S,N))

#einsumを使って行列積を計算する
D=np.einsum("lkmr,krs,lsn->kmn",A,B,C)

print(np.shape(D))

#k=0の要素について検算実施
for l in range(L):
    if l==0:
        D_2=np.matmul(np.matmul(A[l,0,...],B[0,...]),C[l])
    else:
        D_2=D_2+np.matmul(np.matmul(A[l,0,...],B[0,...]),C[l])
        
print("D_2(0)=\n",D_2)
print("D(0)=\n",D[0,...])
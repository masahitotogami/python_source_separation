
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

#ランダムな複素数のテンソル(ndarray)を定義する
b=np.random.uniform(size=L*M)+np.random.uniform(size=L*M)*1.j
b=np.reshape(b,(L,M))

#行列Aのtrace
print("tr(A)= \n",np.trace(A,axis1=-2,axis2=-1))

#einsumを用いたtrace計算
print("tr(A)= \n",np.einsum("lmm->l",A))

#b^H,A,bの計算
print("b^H A b=\n",np.einsum("lm,lmn,ln->l",np.conjugate(b),A,b))
#b^H,A,bの計算
print("trA bb^H =\n",np.trace(np.einsum("lmn,ln,lk->lmk",A,b,np.conjugate(b)),axis1=-2,axis2=-1))



#numpyをインポート（行列を扱う各種関数を含む）
import numpy as np

#A: ...mn
#B: ...ij
#AとBの最後の二軸以外の次元は一致していることを前提とする
def batch_kron(A,B):
 if np.shape(A)[:-2]!=np.shape(B)[:-2]:
     print("error")
     return None
 else:
    return(np.reshape(np.einsum("...mn,...ij->...minj",A,B),np.shape(A)[:-2]+(np.shape(A)[-2]*np.shape(B)[-2],np.shape(A)[-1]*np.shape(B)[-1])))


#乱数の種を設定
np.random.seed(0)



#行列の大きさを設定する
L=10
M=3
R=3
N=3
T=3
#ランダムな複素数のテンソル(ndarray)を定義する
A=np.random.uniform(size=L*M*R)+np.random.uniform(size=L*M*R)*1.j
A=np.reshape(A,(L,M,R))

X=np.random.uniform(size=R*N)+np.random.uniform(size=R*N)*1.j
X=np.reshape(X,(R,N))

B=np.random.uniform(size=L*N*T)+np.random.uniform(size=L*N*T)*1.j
B=np.reshape(B,(L,N,T))


D=np.random.uniform(size=L*M*T)+np.random.uniform(size=L*M*T)*1.j
D=np.reshape(D,(L,M,T))


#1. 多次元配列対応版のクロネッカー積batch_kronの出力結果とnumpyのkronの出力結果が一致していることを確認

#多次元配列対応版のクロネッカー積
C=batch_kron(np.transpose(B,(0,2,1)),A)

#numpyのkronでl=0のデータに対してクロネッカー積を計算
C_2=np.kron(np.transpose(B[0,...],(1,0)),A[0,...])

#多次元配列対応版のクロネッカー積とnumpyのkronとの誤差
print("誤差 = ",np.sum(np.abs(C[0,...]-C_2)))

#2. vecAXBとCvecXが一致しているかどうかを確認する

#Xをベクトル化する
vecX=np.reshape(np.transpose(X,[1,0]),(N*R))

#AXBを計算する
AXB=np.einsum("lmr,rn,lnt->lmt",A,X,B)

#AXBをベクトル化する
vecAXB=np.reshape(np.transpose(AXB,[0,2,1]),(L,T*M))

#CvecX
CvecX=np.einsum("lmr,r->lm",C,vecX)

#vecAXBとCvecXが一致しているかどうかを確認する
print("誤差 = ",np.sum(np.abs(vecAXB-CvecX)))

#3. ΣAXB=ΣDを満たすようなXを求める

#vecDを求める
vecD=np.reshape(np.transpose(D,[0,2,1]),(L,T*M))

#vecXを求める
vecX=np.einsum("mr,r->m",np.linalg.inv(np.sum(C,axis=0)), np.sum(vecD,axis=0))

#Xに戻す
X=np.transpose(np.reshape(vecX,(N,R)),(1,0))

#答えがあっているかどうかを確認
sum_AXB=np.einsum("lmr,rn,lnt->mt",A,X,B)

sum_D=np.sum(D,axis=0)

#sum_AXBとsum_Dが一致しているかどうかを確認する
print("誤差 = ",np.sum(np.abs(sum_AXB-sum_D)))






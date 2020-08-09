
#numpyをインポート（行列を扱う各種関数を含む）
import numpy as np

#行列を定義
a=np.matrix([3.+2.j,1.-1.j,2.+2.j])

#ベクトルを定義
b=np.array([2.+5.j,1.-1.j,4.+1.j])

#ベクトルの内積計算
print("a^Hb=",np.inner(np.conjugate(a),b))

#ベクトルの内積計算
print("a^Ha=",np.inner(np.conjugate(a),a))



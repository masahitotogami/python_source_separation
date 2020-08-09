
#numpyをインポート
import numpy as np

#複素数データを定義する
z=1.0+2.0j
u=2.0+3.0j

#複素数を表示する
print("z=",z)

print("u=",u)

#実部だけを表示する
print("Real(z)=",np.real(z))

#虚部だけを表示する
print("Imaginary(z)=",np.imag(z))

#複素共役を表示する
print("z^*=",np.conjugate(z))

#複素共役を表示する
print("|z|=",np.abs(z))

#zとuを足す
v=z+u
print("z+u=",v)

#zからuを引く
v=z-u
print("z-u=",v)

#zとuをかける
v=z*u
print("z*u=",v)

#zをuで割る
v=z/u
print("z/u=",v)

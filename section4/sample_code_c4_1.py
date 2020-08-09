
#numpyをインポート（行列を扱う各種関数を含む）
import numpy as np

#可視化のためにmatplotlibモジュールをインポート
import matplotlib.pyplot as plt

#アニメーション用のモジュールをインポート
import matplotlib.animation as animation

#確率密度関数の描画用
from scipy.stats import norm



#混合ガウス分布に基づく乱数を生成する
np.random.seed(0)

#各ガウス分布のサンプル数
n_samples=[200,400,400]

#各ガウス分布の平均
means=[-2,3,5]

#各ガウス分布の分散
sigmas=[3,2,0.5]

#GMMに従う乱数を生成
x=None
for n,mean,sigma in zip(n_samples,means,sigmas):
    samples_for_each_dist=np.random.normal(mean,sigma, int(n))
    if x is None:
        x=samples_for_each_dist
    else:
        x=np.concatenate((x,samples_for_each_dist))

#モデルパラメータを初期化する
alpha=np.array([1./3.,1./3.,1./3.])
var=np.array([1.,1.,1.])
mu=np.array([-1,0,1])

#GMMを構成するガウス分布の数
n_clusters=len(alpha)

#繰り返し計算でパラメータを最適化する(ここでは100回繰り返す）
n_iterations=101
log_likelihood=np.zeros(n_iterations)
ims=[]
for t in range(n_iterations):
    print("t{}".format(t))
    if t==0:
        alpha_buf=alpha[None,:]
        var_buf=var[None,:]
        mu_buf=mu[None,:]
    else:
        alpha_buf=np.concatenate((alpha_buf,alpha[None,:]),axis=0)
        var_buf=np.concatenate((var_buf,var[None,:]),axis=0)
        mu_buf=np.concatenate((mu_buf,mu[None,:]),axis=0)
    #Eステップ
    
    #係数部
    coef=alpha/np.sqrt(2.*np.pi*var)
    
    #exponent: n_sample, n_clusters
    exponent=-1.*np.power(x[:,None]-mu[None,:],2.)/(2*var[None,:])

    #βを求める
    beta=coef[None,:]*np.exp(exponent)
    likelihood_each_sample=np.maximum(np.sum(beta,axis=1,keepdims=True),1.e-18)
    beta=beta/likelihood_each_sample

    #対数尤度を求める
    current_log_likelihood=np.average(np.log(likelihood_each_sample))
    log_likelihood[t]=current_log_likelihood

    #Mステップ(パラメータを更新する)
    N=np.maximum(np.sum(beta,axis=0),1.e-18)
    
    #事前確率を更新
    alpha=N/np.sum(N)

    #平均値を更新
    mu=np.einsum("ij,i->j",beta,x)/N
    #分散を計算
    var=np.einsum("ij,ij->j",beta,np.power(x[:,None]-mu[None,:],2.))/N
    var=np.maximum(var,1.e-18)

#対数ゆう度をグラフ化する
plt.figure(figsize=(10,4))
plt.plot(np.arange(0,n_iterations,1),log_likelihood,color="black",linewidth=1,label="log likelihood")
plt.xlabel("Number of iterations")
plt.legend()
plt.savefig("./log_likelihood_gmm.png")
plt.show()


#パラメータ更新の様子をアニメーションで表示
def animation_update(t):
    plt.cla()
    plt.hist(x,bins=50,normed=True,label="observed samples")

    xmin=-20
    xmax=20
    plt.xlim([xmin,xmax])
    plt.ylim([0,0.40])
    x_range=np.arange(-20,20,0.01)

    
    #GMMを描画する
    pdf=None
    for alpha,var, mu in zip(alpha_buf[t],var_buf[t],mu_buf[t]):
        pdf_each=alpha*norm.pdf(x_range,mu,np.sqrt(var))
        if pdf is None:
            pdf=pdf_each
        else:
            pdf=pdf+pdf_each

    plt.plot(x_range,pdf,color="black",linewidth=1,label=r"$p(x|\theta^{(t="+str(t)+")})$")
    plt.legend()

    if t==0:
        plt.savefig("./initialized_gmm.png")
    if t==n_iterations-1:
        plt.savefig("./learned_gmm.png")

#音声データをプロットする
fig=plt.figure(figsize=(10,4))

ani=animation.FuncAnimation(fig,animation_update,interval=200,frames=n_iterations)
plt.show()

#encoding=utf-8
import numpy as np
from matplotlib import pyplot
from numpy import *
from pylab import *
from numpy.linalg import det
import numpy.matlib as ml
import random

def generate_gauss(size):
    mean = [0,10]
    cov = [[1,0],[0,100]]
    x,y = np.random.multivariate_normal(mean,cov,size).T
    return x, y
    
def distmat(X, Y):
    n = len(X)
    m = len(Y)
    xx = ml.sum(X*X, axis=1)
    yy = ml.sum(Y*Y, axis=1)
    xy = ml.dot(X, Y.T)
    return tile(xx, (m, 1)).T+tile(yy, (n, 1)) - 2*xy

def init_params(centers,k):
    pMiu = centers
    pPi = zeros([1,k], dtype=float)
    pSigma = zeros([len(X[0]), len(X[0]), k], dtype=float)
    dist = distmat(X, centers)
    labels = dist.argmin(axis=1)
    for j in range(k):
        idx_j = (labels == j).nonzero()
        pMiu[j] = X[idx_j].mean(axis=0)
        pPi[0, j] = 1.0 * len(X[idx_j]) / N
        pSigma[:, :, j] = cov(mat(X[idx_j]).T)
    return pMiu, pPi, pSigma
    
def calc_prob(k,pMiu,pSigma):
    Px = zeros([N, k], dtype=float)
    for i in range(k):
        Xshift = mat(X - pMiu[i, :])
        inv_pSigma = mat(pSigma[:, :, i]).I
        coef = pow((2*pi), (len(X[0])/2)) * sqrt(det(mat(pSigma[:, :, i])))            
        for j in range(N):
            tmp = (Xshift[j, :] * inv_pSigma * Xshift[j, :].T)
            Px[j, i] = 1.0 / coef * exp(-0.5*tmp)
    return Px

def MoG(X, k, threshold=1e-15):
    N = len(X)
    labels = zeros(N, dtype=int)
    centers = array(random.sample(X, k))
    iter = 0
    pMiu, pPi, pSigma = init_params(centers,k)
    Lprev = float('-10000')
    pre_esp = 100000
    while iter < 100:
        Px = calc_prob(k,pMiu,pSigma)
        pGamma =mat(array(Px) * array(pPi))
        pGamma = pGamma / pGamma.sum(axis=1)
        Nk = pGamma.sum(axis=0)
        pMiu = diagflat(1/Nk) * pGamma.T * mat(X)
        pPi = Nk / N
        pSigma = zeros([len(X[0]), len(X[0]), k], dtype=float)
        for j in range(k):
            Xshift = mat(X) - pMiu[j, :]
            for i in range(N):
                pSigmaK = Xshift[i, :].T * Xshift[i, :]
                pSigmaK = pSigmaK * pGamma[i, j] / Nk[0, j]
                pSigma[:, :, j] = pSigma[:, :, j] + pSigmaK
        labels = pGamma.argmax(axis=1)
        iter = iter + 1
        L = sum(log(mat(Px) * mat(pPi).T))
        cur_esp = L-Lprev
        if cur_esp < threshold:
            break
        if cur_esp > pre_esp:
            break
        pre_esp=cur_esp
        Lprev = L
        print "iter %d esp %lf" % (iter,cur_esp)
    pylab_plot(X, labels,iter)

def pylab_plot(X, labels,iter):
    colors = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pyplot.plot(hold=False)
    pyplot.hold(True)
    labels = array(labels).ravel()
    data_colors=[colors[lbl] for lbl in labels]
    pyplot.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
    pyplot.savefig('iter_%02d.png' % iter, format='png')

samples = generate_gauss(1000)
N = len(samples[0])
X = zeros((N, 2))
for i in range(N):
   X[i, 0] = samples[0][i]
   X[i, 1] = samples[1][i]
MoG(X, 3)
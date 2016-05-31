#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt

sample = []
#读取文件
file = open('optdigits.tra')
for line in file:
    list = map(int, line.rstrip().split(','))
    if list[-1] == 3:
        list = list[:-1]
        sample.append(list)
sample = np.mat(sample).T
mean = np.mean(sample, axis=1)
sample = sample - mean
#pca
cov_mat = np.cov(sample,rowvar=1)
eigVals, eigVects = np.linalg.eig(cov_mat)
eigValIndice = np.argsort(eigVals)
n_eigValIndice = eigValIndice[-1:-3:-1]
n_eigVect = eigVects[:,n_eigValIndice]
low_data = eigVects.T * sample

figure2 = plt.figure(num=2)
xlist = []
ylist = []
for i in range(low_data[0].size):
    x = int(low_data[:, i][0])/5
    y = int(low_data[:, i][1])/5
    xlist.append(x)
    ylist.append(y)
xlist = np.array(xlist)
ylist = np.array(ylist)
#绘制
plt.plot(xlist, ylist, 'o', color='green')
plt.xlabel('First Component')
plt.ylabel('Second Component')
plt.xlim(-6, 6)
plt.ylim(-8, 8)
plt.grid()
plt.show()

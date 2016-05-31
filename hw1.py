import numpy as np
import scipy
import matplotlib.pyplot as plt

#生成sin(x)曲线的点
x1 = np.linspace(0, 1, 1000)
y1 = np.sin(2.0*x1*np.pi)

#生成10个高斯噪声样本点
x2 = np.linspace(0,1,10)
noise = np.random.normal(0, 0.2, size=10)
y2 = np.sin(2.0*x2*np.pi) + noise

#进行3次方的曲线拟合
#得到相应的拟合曲线参数
p = scipy.polyfit(x2,y2,3)
#生成一个参数为weight的多项式
f = np.poly1d(p)
#绘制
fig = plt.figure('curve fitting')
#绘制坐标轴
plt.xlabel('x')
plt.ylabel('t')
plt.ylim(-1.5, 1.5)
#绘制sin曲线
plt.plot(x1, y1, color='green', linewidth=2)
#绘制噪声点
plt.plot(x2, y2, 'o', mew=2, mec='b', mfc='none', ms=6)
#绘制拟合曲线
x = np.linspace(0, 1, 1000)
plt.plot(x, f(x), '-r', linewidth=2)
plt.title('degree 3 in 10 samples')
plt.show()

#对10个样本点进行9次方的曲线拟合
#得到相应的拟合曲线参数
p = scipy.polyfit(x2,y2,9)
#生成一个参数为weight的多项式
f = np.poly1d(p)
#绘制
fig = plt.figure('curve fitting')
#绘制坐标轴
plt.xlabel('x')
plt.ylabel('t')
plt.ylim(-1.5, 1.5)
#绘制sin曲线
plt.plot(x1, y1, color='green', linewidth=2)
#绘制噪声点
plt.plot(x2, y2, 'o', mew=2, mec='b', mfc='none', ms=6)
#绘制拟合曲线
x = np.linspace(0, 1, 1000)
plt.plot(x, f(x), '-r', linewidth=2)
plt.title('degree 9 in 10 samples')
plt.show()

#对15个样本点进行9次方的曲线拟合
#生成15个样本点
x2 = np.linspace(0,1,15)
noise = np.random.normal(0, 0.2, size=15)
y2 = np.sin(2.0*x2*np.pi) + noise
#得到相应的拟合曲线参数
p = scipy.polyfit(x2,y2,9)
#生成一个参数为weight的多项式
f = np.poly1d(p)
#绘制
fig = plt.figure('curve fitting')
#绘制坐标轴
plt.xlabel('x')
plt.ylabel('t')
plt.ylim(-1.5, 1.5)
#绘制sin曲线
plt.plot(x1, y1, color='green', linewidth=2)
#绘制噪声点
plt.plot(x2, y2, 'o', mew=2, mec='b', mfc='none', ms=6)
#绘制拟合曲线
x = np.linspace(0, 1, 1000)
plt.plot(x, f(x), '-r', linewidth=2)
plt.title('degree 9 in 15 samples')
plt.show()

#对100个样本点进行9次方的曲线拟合
#生成100个样本点
x2 = np.linspace(0,1,100)
noise = np.random.normal(0, 0.2, size=100)
y2 = np.sin(2.0*x2*np.pi) + noise
#得到相应的拟合曲线参数
p = scipy.polyfit(x2,y2,9)
#生成一个参数为weight的多项式
f = np.poly1d(p)
#绘制
fig = plt.figure('curve fitting')
#绘制坐标轴
plt.xlabel('x')
plt.ylabel('t')
plt.ylim(-1.5, 1.5)
#绘制sin曲线
plt.plot(x1, y1, color='green', linewidth=2)
#绘制噪声点
plt.plot(x2, y2, 'o', mew=2, mec='b', mfc='none', ms=6)
#绘制拟合曲线
x = np.linspace(0, 1, 1000)
plt.plot(x, f(x), '-r', linewidth=2)
plt.title('degree 9 in 100 samples')
plt.show()

#对10个样本点进行含正则项的9次方曲线拟合
#生成10个样本点
x2 = np.linspace(0,1,10)
noise = np.random.normal(0, 0.2, size=10)
y2 = np.sin(2.0*x2*np.pi) + noise

data = np.matrix(np.zeros(shape=(10, x2.size)))
for i in range(0, 10):
   value = x2**(9-i)
   data[i, :] = value
#使用正规方程计算参数值
p = np.linalg.inv(data*data.T + np.exp(-18)*np.eye(10)) * data * np.matrix(y2).T
p = p.A1
#生成一个参数为weight的多项式
f = np.poly1d(p)
#绘制
fig = plt.figure('curve fitting')
#绘制坐标轴
plt.xlabel('x')
plt.ylabel('t')
plt.ylim(-1.5, 1.5)
#绘制sin曲线
plt.plot(x1, y1, color='green', linewidth=2)
#绘制噪声点
plt.plot(x2, y2, 'o', mew=2, mec='b', mfc='none', ms=6)
#绘制拟合曲线
x = np.linspace(0, 1, 1000)
plt.plot(x, f(x), '-r', linewidth=2)
plt.title('degree 9 in 10 samples with regularization')
plt.show()
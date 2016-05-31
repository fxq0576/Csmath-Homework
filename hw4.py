# -*- coding: utf-8 -*-
from pylab import *
from numpy import *
from math import *

def vec_inner(v):
    return sum(v * v);

#初始数据
data_1= [1.2,1.8,2.0,1.5,5.0,3.0,4.0,6.0,8.0]
obs_1 = [19.21,18.15,15.36,14.10,12.89,9.32,7.45,5.24,3.01]
#初始参数
a0=10.0
b0=0.5
y_init = zeros(len(data_1))
for x_i in range(0,len(data_1)):
    y_init[x_i] = a0*exp(-b0*data_1[x_i])
Ndata = len(obs_1)
Nparams = 2
n_iters = 50
lamda = 0.01
#变量赋值
updateJ = 1
a_est = a0
b_est = b0
#迭代
i = 0
y_est = zeros(len(data_1))
y_est_lm = zeros(len(data_1))
while i < n_iters:
    if updateJ == 1:
#计算雅克比矩阵
        J=zeros([Ndata,Nparams])
        for j in range(0,len(data_1)):
            J[j][0] = exp(-b_est*data_1[j])
            J[j][1] = -a_est*data_1[j]*exp(-b_est*data_1[j])
        for y_i in range(len(data_1)):
            y_est[y_i] = a_est*exp(-b_est*data_1[y_i])
        d = obs_1 - y_est
#计算hessen矩阵
        H = dot(J.T,J)
        if i == 0:
            e = vec_inner(d)
    H_lm = H + (lamda*eye(Nparams))
    g = dot(J.T,d)
    dp = dot(inv(H_lm),g)
    a_lm = a_est+dp[0]
    b_lm = b_est+dp[1]
    for y_lm_i in range(len(data_1)):
        y_est_lm[y_lm_i] = a_lm*exp(-b_lm*data_1[y_lm_i])
    d_lm = obs_1-y_est_lm
    e_lm = vec_inner(d_lm)
    if e_lm < e:
        lamda = lamda/10
        a_est = a_lm
        b_est = b_lm
        e = e_lm
        updateJ = 1
    else:
        updateJ = 0
        lamda = lamda*10
    i = i + 1
    print "%f\t%f" %(a_est, b_est)
print "best_parameter_1=%f; best_parameter_2=%f" %(a_est, b_est)

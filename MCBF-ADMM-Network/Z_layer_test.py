import cvxpy as cvx
import numpy as np
import scipy as sp


#常数设置
Nc=2 #基站数量
k=2 #移动台数量
Nt = 8 #天线数
gamma = cvx.Parameter()
gamma.value = 10 #设定信噪比阈值γ
epsilon = 0.05 #设定球形误差模型的误差半径ε
I=np.eye(Nt)  #生成维度为Nt单位矩阵
Q = cvx.Parameter((Nt,Nt))
Q.value=(np.square(np.true_divide(1,epsilon)))*I
sigma = cvx.Parameter()
sigma = 0.02 #噪声功率



h111=np.random.rand(8,1)
h112=np.random.rand(8,1)
h221=np.random.rand(8,1)
h222=np.random.rand(8,1)
h121=np.random.rand(8,1)
h122=np.random.rand(8,1)
h211=np.random.rand(8,1)
h212=np.random.rand(8,1)

E1 = np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])

#外部参数
c = cvx.Parameter()
c = 1

t = cvx.Parameter((4,1))
t=np.array([[0],[0],[0],[0]])
v1=np.array([0, 0 ,0 ,0])#参数v1 记得转置


#定义优化变量

W11 = cvx.Variable((Nt,Nt),PSD = True)
W12 = cvx.Variable((Nt,Nt),PSD = True)
lambda111 = cvx.Variable()
lambda112 = cvx.Variable()
lambda121 = cvx.Variable()
lambda122 = cvx.Variable()
# t121 = cvx.Variable()
# t122 = cvx.Variable()
# t211 = cvx.Variable()
# t212 = cvx.Variable()
t1 = cvx.Variable((4,1))

# 定义约束条件

# h111t=np.transpose(h111)
# h112t=np.transpose(h112)
# h221t=np.transpose(h221)
# h222t=np.transpose(h222)
# h121t=np.transpose(h121)
# h122t=np.transpose(h122)
# h211t=np.transpose(h211)
# h212t=np.transpose(h212)

h111tc=cvx.conj(h111).T
h112tc=cvx.conj(h112).T
h221tc=cvx.conj(h221).T
h222tc=cvx.conj(h222).T
h121tc=cvx.conj(h121).T
h122tc=cvx.conj(h122).T
h211tc=cvx.conj(h211).T
h212tc=cvx.conj(h212).T


phi11 =  cvx.bmat([[(1/gamma)*W11-W12+lambda111*Q, ((1/gamma)*W11-W12)@ h111 ],
[h111tc * ((1/gamma) * W11-W12), h111tc @ ((1/gamma)*W11-W12)@h111-lambda111-t1[2]-sigma*sigma]])

phi12 = cvx.bmat([[(1/gamma)*W12-W11+lambda112*Q, ((1/gamma)*W12-W11)@ h112 ],
[h112tc * ((1/gamma) * W12-W11), h112tc @ ((1/gamma)*W12-W11)@h112-lambda112-t1[3]-sigma*sigma]])

psi121 = cvx.bmat([[-(W11+W12)+lambda121*Q, -(W11+W12)@h121],
[-h121tc@(W11+W12),-h121tc@(W11+W12)@h121+t1[0]-lambda121]])

psi122= cvx.bmat([[-(W11+W12)+lambda122*Q,-(W11+W12)@h122],
[-h122tc@(W11+W12),-h122tc@(W11+W12)@h122+t1[1]-lambda122]])



# phi111 = cvx.conj(cvx.conj(phi11))
# phi121 = cvx.conj(cvx.conj(phi12))
# psi1211 = cvx.conj(cvx.conj(psi121))
# psi1221 = cvx.conj(cvx.conj(psi122))
phi111 = cvx.constraints.PSD(phi11)
phi121 = cvx.constraints.PSD(phi12)
psi1211 = cvx.constraints.PSD(psi121)
psi1221 = cvx.constraints.PSD(psi122)

constraints = [phi111, phi121,
               psi1211,psi1221,
               lambda111>=0,        lambda112>=0,
                lambda121>=0,        lambda122>=0,
                W11>>0,        W12>>0,
                t1>=0
                # t121>=0 ,
                # t122>=0 ,
                # t211>=0 ,
                # t212>=0 ,
               ]

# constraints = [
#                lambda111>=0,        lambda112>=0,
#                 lambda121>=0,        lambda122>=0,
#                 W11>>0,        W12>>0,
#
#                 t121>=0 ,
#                 t122>=0 ,
#                 t211>=0 ,
#                 t212>=0 ,
#                ]

# 定义优化问题
E1t = E1@t
#np.array([[E1t[0]-t211],[E1t[1]-t212],[E1t[2]-t121],[E1t[3]-t122]])

obj = cvx.Minimize(cvx.trace(W11)+cvx.trace(W12) - (v1 @ (E1t-t1 )))
#          +(c/2)*cvx.square(cvx.norm( E1t - t111 )))
# 定义优化问题
prob = cvx.Problem(obj, constraints)
#求解问题 solver=cvx.CVXOPT
prob.solve()                      #返回最优值
print("status:", prob.status)     #求解状态
print("optimal value", prob.value) #目标函数优化值


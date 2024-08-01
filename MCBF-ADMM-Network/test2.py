# import matlab.engine
# global eng
# eng = matlab.engine.start_matlab()
#
#
# H = eng.H(8,8)
#
# v1 = matlab.double([[0],[0],[0],[0]])
# t = matlab.double([[0],[0],[0],[0]])
# ret = eng.Z1(0.0,0.0,v1,t,H,nargout=10)
#
# print(ret[0])


import torch
import matlab.engine
import scipy.io as sio

E = torch.Tensor([[0, 0, 0.5, 0, 0.5, 0, 0, 0], [0, 0, 0, 0.5, 0, 0.5, 0, 0], [0.5, 0, 0, 0, 0, 0, 0.5, 0],  [0, 0.5, 0, 0, 0, 0, 0, 0.5]])
print

E1t= E.tolist()
E2 = matlab.double(E1t)
t211 = torch.tensor([1])
t212 = torch.tensor([1])
t121 = torch.tensor([1])
t122 = torch.tensor([1])




t1 = torch.Tensor([[t211], [t212], [t121], [t122]])
t2 = torch.Tensor([t121, t122, t211, t212]).T


H =0.707*torch.randn((10,8,8))
v2 = torch.zeros(10,4,1)

rho1 = torch.zeros(1)
rho2 = rho1.item()


A = torch.tensor([1.],requires_grad=True)


# H1 = torch.tensor ([[-0.395397041799618,	0.356324713677101,	-0.125787286706314,	-0.910340522898403,	0.0974356839706305	,-0.580625359964030,	-0.169551269248824	,-0.992859826428695],
# [-0.571939895117,   -0.581581577007,	1.25333443909,	0.154209278094,	-1.12205217240,- 0.757951328546	,0.651235059763	,0.145355384026574],
# [0.820954003331,	0.142115643629	,-1.77399204747	,-1.10773020099	,-0.720644704557	,-0.759497479698,-0.641261972403	,-1.4079880453],
# [0.418681650989804,	-0.712093671212722	,-0.322857309183956	,0.553863964875646	,-0.979508171807843	,0.614866686651010	,0.102134932176081,	0.0200881498198891],
# [0.171648519674210,	-0.433577768272249	,1.71853643721188	,-0.219606088489509	,0.675206670133200	,0.693708107903467	,0.987410293137466,	1.36508053671389],
# [0.170043260139469,	-0.465957482223376	,-0.333368381157086	,0.463384335300251	,-0.425056254136229	,-1.24367734544113,	-0.134137279299292	,0.665484205049906],
# [-0.580888502772182,	-0.589182107767211	,-0.396212075783006	,-0.380046814689614	,-0.828651934421281	,-0.104719655861968	,-0.596682080611657,	-0.00489632647154977],
# [0.702236318923942	,0.267413145333260	,-0.867243396892325	,0.232385979032570	,-0.408078611640754	,0.178149717184599	,1.37421822386300,	-0.239172962465960]])

H_temp1 = sio.loadmat('D:\Python36/data_r.mat')
H_temp2 = sio.loadmat('D:\Python36/data_i.mat')
H1 = H_temp1["data_r"]
H2 = H_temp2["data_i"]
H1=torch.from_numpy(H1)
H4=H1[0]
H3 = H1.shape[0]

sum=torch.zeros(200,1)
sum1= sum[13,0]
#W21 = torch.zeros(200,8, 8)

X_temp1 = sio.loadmat('D:\Python36/Z_learn/Z_in_save.mat')
Y_temp1 = sio.loadmat('D:\Python36/Z_learn/Z_out_save.mat')
Xtrain = X_temp1["Z_in_save"]
Ytrain = Y_temp1["Z_out_save"]
m=70
A0=torch.zeros([m,m],dtype=torch.double)
#X1=Xtrain[1]

P_out_temp = sio.loadmat('D:\Python36/P_out.mat')
P_out = P_out_temp["P_out"]

P1_out = P_out[0]
P2_out = P_out[1]

grad_v2 = grad_t = torch.tensor([[0], [0], [0], [0]])
grad_t[2][0] =1.0
grad_t2=None

temp1 = sio.loadmat('D:\Python36/P_final_out.mat')
P_final_out = temp1["P_final_sum"]
P1=P_final_out[0]
P2=P_final_out[1]
P3=P_final_out[2]
P4=P_final_out[3]
P5=P_final_out[4]
P6=P_final_out[5]
P7=P_final_out[6]

H10=H1[0,:,:]
H10[0,0]=1;
h111=H10[:,0]

h1111=h111[0]
w11=torch.tensor([[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]])
w12=torch.tensor([[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]])

gamma1=torch.tensor(10.)
sigma1=torch.tensor(0.02)
t211=torch.tensor(0.)
w11relu=torch.relu(w11)
w12relu=torch.relu(w12)
W11=w11relu*w11relu.T
W12=w12relu*w12relu.T

h111=h111.view(8,1).double()

phi111=(1/gamma1)*W11-W12
phi112=torch.matmul((h111.T),((1/gamma1)*W11-W12).double())
       # *h111.view(8,1)-t211-sigma1*sigma1

b = torch.zeros([85, 1], dtype=torch.double)


x2=H10[0:2,0:2]
n2=torch.tensor(70.).double()
x3=1/torch.sqrt(n2).double()
X = torch.eye(70).double()
normsA = torch.norm(X)
normsB = torch.norm(x2)
x4=torch.max(normsB,torch.sqrt(normsA))


AA=torch.tensor([1.,2.,3.])

AA2=torch.reshape(AA,(3,1))

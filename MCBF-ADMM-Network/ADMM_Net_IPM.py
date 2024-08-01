import torch
import matlab.engine
from Z1_layer import Z1_Layer
from Z2_layer import Z2_Layer
from Xt_layer import Xt_Layer
from Xrho1_layer import  Xrho1_Layer
from Xrho2_layer import  Xrho2_Layer
from Mv1_layer import Mv1_Layer
from Mv2_layer import Mv2_Layer
from Mmu1_layer import Mmu1_Layer
from Mmu2_layer import Mmu2_Layer
from ZXM_IPM_layer import ZXM_layer  # 自定义层
import mbegin
import Mydataset
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import scipy.io as sio
import torch.optim as optim
import numpy as np


N, D_in, D_out = 10, 5, 3  # 一共10组样本，输入特征为5，输出特征为3
sum=torch.zeros(200,1)


# 先定义一个模型
class ADMM_Net(torch.nn.Module):
    def __init__(self):
        super(ADMM_Net, self).__init__()  # 第一句话，调用父类的构造函数
        self.ZXM_layer = ZXM_layer()
        c = torch.tensor([1.0], requires_grad=True)
        # iter = 10;c=15854 0.12;
        # iter = 20;c=4850 0.08 ok
        # iter = 30;c=4348-4350 0.05 ok
        # iter = 50;c=2692-2695 0.05
        # iter = 80;c=4853-4901 0.04
        # iter = 100;c=4991-5004 0.03 5159-5250
        c = c.double()
        self.c = torch.nn.Parameter(c, requires_grad=True)  # 由于weights是可以训练的，所以使用Parameter来定义
    def forward(self, rho1,rho2,mu1,mu2,v1,v2,t,H1,H2,W11,W12,W21,W22,P1_out,P2_out):

        for stage in range(99):
            #i
            print('stage: {}'.format(stage+1))
            rho1,rho2,mu1,mu2,v1,v2,t,H1,H2,p1,p2 = self.ZXM_layer(rho1,rho2,mu1,mu2,v1,v2,t,H1,H2,self.c)

            p11 = p1.item()
            p22 = p2.item()
            sum11 = (p11 + p22)
            print("功率为 %.10f  " % (sum11))
            # loss1 = (p11 - P1_out).norm(2) / P1_out + (p22 - P2_out).norm(2) / P2_out
            loss1 = (p11+p22 - P1_out-P2_out).norm(2) / (P1_out + P2_out)
            print('Loss: {}'.format(loss1))



        return p1,p2



model = ADMM_Net()
#model.apply(weigth_init)

print(model)

for para1 in model.named_parameters():
    print(para1)

mbegin.eng = matlab.engine.start_matlab()


trainDataset = Mydataset.Mydataset()
testDataset = Mydataset.Mydataset_test()

train_loader = torch.utils.data.DataLoader(dataset=trainDataset,
                          batch_size=1,
                          shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=testDataset,
                          batch_size=1,
                          shuffle=True)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)   # 学习率为0.001
loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差处理回归)
losses = []#记录每次迭代后训练的loss
eval_losses = []#测试的

# for epoch in range(1):
#     running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
#     for i, data in enumerate(train_loader2):
#         # 将数据从 train_loader 中读出来,一次读取的样本数是32个
#         rho1,rho2,mu1,mu2,v1,v2,t,H1,H2,W11,W12,W21,W22,P1_out,P2_out = data
#         rho1, rho2, mu1, mu2, v1, v2, t, H1, H2, W11, W12, W21, W22=rho1[0],rho2[0],mu1[0],mu2[0],v1[0],v2[0],t[0],H1[0],H2[0],W11[0],W12[0],W21[0],W22[0]
#
#         # 将这些数据转换成Variable类型
#         #inputs, labels = Variable(rho1), Variable(rho2)
#
#         optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
#
#         yp1,yp2 = model(rho1, rho2, mu1, mu2, v1, v2,t, H1, H2, W11, W12, W21, W22)
#
#         p11 = yp1.item()
#         p22 = yp2.item()
#         #sum[i,0]= 10 * np.log10(1000 * (p11 + p22))
#         #sum1 = sum[i,0]
#         sum11=1*(p11+p22)
#         print("功率为 %.10f  " %(sum11))
#         loss = loss_func(yp1, P1_out)+loss_func(yp2, P2_out)
#         optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
#         loss.backward()
#         optimizer.step()
#
#     losses.append(loss.item())


for epoch in range(1):

    #for i, data in enumerate(train_loader2):
    for i in range(5):
        train_loss = 0
        model.train()  # 网络设置为训练模式 暂时可加可不加
        for rho1,rho2,mu1,mu2,v1,v2,t,H1,H2,W11,W12,W21,W22,P1_out,P2_out in train_loader:

            # 将数据从 train_loader 中读出来,一次读取的样本数是10个

            rho1, rho2, mu1, mu2, v1, v2, t, H1, H2, W11, W12, W21, W22,P1_out,P2_out=rho1[0],rho2[0],mu1[0],mu2[0],v1[0],v2[0],t[0],H1[0],H2[0],W11[0],W12[0],W21[0],W22[0],P1_out[0],P2_out[0]

            # 将这些数据转换成Variable类型
            #inputs, labels = Variable(rho1), Variable(rho2)

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            yp1,yp2 = model(rho1, rho2, mu1, mu2, v1, v2,t, H1, H2, W11, W12, W21, W22,P1_out,P2_out)

            p11 = yp1.item()
            p22 = yp2.item()
            #sum[i,0]= 10 * np.log10(1000 * (p11 + p22))
            #sum1 = sum[i,0]
            sum11=1*(p11+p22)
            print("功率为 %.10f  " %(sum11))

            loss = loss_func(yp1, P1_out)/P1_out+loss_func(yp2, P2_out)/P2_out
            # loss = (yp1- P1_out).norm(2) / P1_out + (yp2-P2_out).norm(2) / P2_out
            # loss = (yp1+yp2- P1_out-P2_out).norm(2) / (P1_out+ P2_out)
            # loss = loss_func((yp1 + yp2 ), (P1_out + P2_out))/(P1_out + P2_out)
            # optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            loss.backward()
            optimizer.step()

            for para1 in model.named_parameters():
                print(para1)
            print('Loss: {}'.format(loss))
            train_loss = train_loss + loss.item()

        losses.append(train_loss / len(train_loader))

        print('betch: {}, trainloss: {}'.format(i , train_loss / len(train_loader),
                                                          ))

    # 测试集进行测试
    eval_loss = 0
    model.eval()  # 可加可不加
    for rho1,rho2,mu1,mu2,v1,v2,t,H1,H2,W11,W12,W21,W22,P1_out,P2_out in test_loader:
        # 前向传播
        # 将数据从 train_loader 中读出来,一次读取的样本数是10个

        rho1, rho2, mu1, mu2, v1, v2, t, H1, H2, W11, W12, W21, W22, P1_out, P2_out = rho1[0], rho2[0], mu1[0], mu2[0], \
                                                                                      v1[0], v2[0], t[0], H1[0], H2[0], \
                                                                                      W11[0], W12[0], W21[0], W22[0], \
                                                                                      P1_out[0], P2_out[0]

        yp1, yp2 = model(rho1, rho2, mu1, mu2, v1, v2, t, H1, H2, W11, W12, W21, W22,P1_out,P2_out)
        p11 = yp1.item()
        p22 = yp2.item()
        # 记录单批次一次batch的loss，测试集就不需要反向传播更新网络了
        loss = (p11 + p22 - P1_out - P2_out).norm(2) / (P1_out + P2_out)
        # loss = loss_func(yp1, P1_out)+loss_func(yp2, P2_out)
        eval_loss = eval_loss + loss.item()

    eval_losses.append(eval_loss / len(test_loader))

    print('epoch: {}， evalloss: {}'.format(epoch,  eval_loss / len(test_loader)))


#for k in range(10):


# for name,parameters in model.named_parameters():
#     print(name,':',parameters.size(),parameters)

import torch
import numpy as np

class Xt_Layer(torch.nn.Module):
    '''
    因为这个层实现的功能是：
     % X层 原文
        t=(inv(E'*E)*E')*([t1.' t2.'].'-(1/c)*[v1.' v2.'].');
        rho1=p1-(1/c)*mu1;
        rho2=p2-(1/c)*mu2;
        T{j}=t;
        RHO1{j}=rho1;
        RHO2{j}=rho2;
    所以有一个参数：c

    输入 t1 t2 v1 v2 的维度是（4*1)
    输出 t 的维度是（4*1)
    '''

    def __init__(self):
        super(Xt_Layer, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        # self.in_features = in_features
        # self.out_features = out_features
        #c = torch.tensor([0.000001], requires_grad=True)
        #self.c = torch.nn.Parameter(c,requires_grad=True)  # 由于weights是可以训练的，所以使用Parameter来定义
        # if bias:
        #     self.bias = torch.nn.Parameter(torch.Tensor(in_features))  # 由于bias是可以训练的，所以使用Parameter来定义
        # else:
        #     self.register_parameter('bias', None)

    def forward(self, t1,t2,v1,v2,c):
        E = torch.Tensor([[0, 0, 0.5, 0, 0.5, 0, 0, 0], [0, 0, 0, 0.5, 0, 0.5, 0, 0], [0.5, 0, 0, 0, 0, 0, 0.5, 0],  [0, 0.5, 0, 0, 0, 0, 0, 0.5]]).double()

        t3 = torch.transpose(t1, 0, 1)
        t4 = torch.transpose(t2, 0, 1)
        t5 = torch.cat((t3,t4),1)
        t6 = torch.transpose(t5, 0, 1)
        v3 = torch.transpose(v1, 0, 1)
        v4 = torch.transpose(v2, 0, 1)
        v5 = torch.cat((v3, v4), 1)
        v6 = torch.transpose(v5, 0, 1)
        c2 = torch.div(1,c)
        t = torch.matmul(E, (t6 - c2*v6))
        return t


import torch


class Xrho2_Layer(torch.nn.Module):
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

    输入p,mu 的维度是（1*1)
    输出 rho 的维度是（1*1)
    '''

    def __init__(self):
        super(Xrho2_Layer, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        # self.in_features = in_features
        # self.out_features = out_features
        # c = torch.tensor([0.000001], requires_grad=True)
        # self.c = torch.nn.Parameter(c, requires_grad=True)  # 由于weights是可以训练的，所以使用Parameter来定义
        # if bias:
        #     self.bias = torch.nn.Parameter(torch.Tensor(in_features))  # 由于bias是可以训练的，所以使用Parameter来定义
        # else:
        #     self.register_parameter('bias', None)

    def forward(self, p2,mu2,c):

        c2 = torch.div(1, c)
        rho2 = p2 - c2 * mu2



        return rho2
import torch


class Mv2_Layer(torch.nn.Module):
    '''
    因为这个层实现的功能是：
     % M层
        v1=v1+lamda*(E1*t-t1);
        mu1=mu1+lamda*(rho1-p1);
        v2=v2+lamda*(E2*t-t2);
        mu2=mu2+lamda*(rho2-p2);
    所以有一个个参数：lamda(lambda被占用)

    输入 t,t1 v1的维度是（4*1)
    输出 v1 的维度是（4*1)
    '''

    def __init__(self):
        super(Mv2_Layer, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        # self.in_features = in_features
        # self.out_features = out_features
        # lamda = torch.tensor([0.000001], requires_grad=True)
        # self.lamda = torch.nn.Parameter(lamda, requires_grad=True)  # 由于weights是可以训练的，所以使用Parameter来定义
        # if bias:
        #     self.bias = torch.nn.Parameter(torch.Tensor(in_features))  # 由于bias是可以训练的，所以使用Parameter来定义
        # else:
        #     self.register_parameter('bias', None)

    def forward(self, t,t2,v2,lamda):

        E2 = torch.Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).double()
        E2t = torch.matmul(E2,t)

        v2 = v2 + lamda * (E2t-t2)


        return v2
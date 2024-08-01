import torch


class Mmu2_Layer(torch.nn.Module):
    '''
    因为这个层实现的功能是：
     % M层
        v1=v1+lamda*(E1*t-t1);
        mu1=mu1+lamda*(rho1-p1);
        v2=v2+lamda*(E2*t-t2);
        mu2=mu2+lamda*(rho2-p2);
    所以有一个个参数：lamda(lambda被占用)

    输入 rho1 , p1 , mu1的维度是（1*1)
    输出 mu1 的维度是（1*1)
    '''

    def __init__(self):
        super(Mmu2_Layer, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        # self.in_features = in_features
        # self.out_features = out_features
        # lamda = torch.tensor([0.000001], requires_grad=True)
        # self.lamda = torch.nn.Parameter(lamda, requires_grad=True)  # 由于weights是可以训练的，所以使用Parameter来定义
        # if bias:
        #     self.bias = torch.nn.Parameter(torch.Tensor(in_features))  # 由于bias是可以训练的，所以使用Parameter来定义
        # else:
        #     self.register_parameter('bias', None)

    def forward(self, rho2, p2, mu2,lamda):

        mu2 = mu2 + lamda * (rho2 - p2)

        return mu2
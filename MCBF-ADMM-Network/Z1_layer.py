import torch
import matlab.engine
import mbegin


import torch


class Z1_function(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx,rho1,mu1,v1,t,H1,H2,c):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(rho1,mu1,v1,t,c)
        rho1 = rho1.item()
        mu1 = mu1.item()
        v1 = matlab.double(v1.tolist())
        t = matlab.double(t.tolist())
        H1 = matlab.double(H1.tolist())
        H2 = matlab.double(H2.tolist())
        c = c.item()
        ret = mbegin.eng.Z1(rho1, mu1, v1, t, H1, H2 , c, nargout=10)
        W11 = torch.tensor(ret[0]).double()
        W12 = torch.tensor(ret[1]).double()
        lambda111 = torch.tensor(ret[2]).double()
        lambda112 = torch.tensor(ret[3]).double()
        lambda121 = torch.tensor(ret[4]).double()
        lambda122 = torch.tensor(ret[5]).double()
        t121 = torch.tensor(ret[6]).double()
        t122 = torch.tensor(ret[7]).double()
        t211 = torch.tensor(ret[8]).double()
        t212 = torch.tensor(ret[9]).double()
        p1 = (torch.trace(W11) + torch.trace(W12)).double()

        return p1,W11,W12,t121,t122,t211,t212

    @staticmethod
    def backward(ctx, grad_p1,grad_W11,grad_W12,grad_t121,grad_t122,grad_t211,grad_t212):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        rho1,mu1,v1,t,c= ctx.saved_tensors

        grad_rho1=grad_mu1=grad_H1=grad_H2=grad_c=None
        grad_v1 = grad_t = torch.tensor([[0],[0],[0],[0]]).double()

        h_ = torch.tensor(0.001).double()
        n_ = torch.tensor(10.0).double()
        if ctx.needs_input_grad[0]:
            grad_rho1 = grad_p1/(torch.pow((1+h_*2*rho1),n_))
        if ctx.needs_input_grad[1]:
            grad_mu1 = None
        if ctx.needs_input_grad[2]:
            grad_v1[2][0] = grad_t121/(torch.pow( ( 1+h_), n_))
            grad_v1[3][0] = grad_t122 / (torch.pow((1 + h_), n_))
            # grad_v1 = (grad_t121 / (torch.pow((1 + h_), n_))+grad_t122 / (torch.pow((1 + h_), n_)))/2
        if ctx.needs_input_grad[3]:
            grad_t[0][0] = (grad_p1/(torch.pow((1+h_), n_)) + grad_t121/(torch.pow((1+h_*(1+2*t[0])), n_))+grad_t122/(torch.pow((1+h_), n_))) /3
            grad_t[1][0] = (grad_p1 / (torch.pow((1 + h_), n_)) + grad_t121 / (torch.pow((1 + h_ ), n_)) + grad_t122 / (torch.pow((1 + h_* (1 + 2 * t[1])), n_))) / 3
            grad_t[2][0] = (grad_p1 / (torch.pow((1 + h_), n_)) + grad_t121 / (torch.pow((1 + h_), n_)) + grad_t122 / (torch.pow((1 + h_ ), n_))) / 3
            grad_t[3][0] = (grad_p1 / (torch.pow((1 + h_), n_)) + grad_t121 / (torch.pow((1 + h_), n_)) + grad_t122 / (torch.pow((1 + h_), n_))) / 3
            # grad_t = ((grad_p1/(torch.pow((1+h_), n_)) + grad_t121/(torch.pow((1+h_*(1+2*t[0])), n_))+grad_t122/(torch.pow((1+h_), n_))) /3 +
            #           (grad_p1 / (torch.pow((1 + h_), n_)) + grad_t121 / (torch.pow((1 + h_), n_)) + grad_t122 / (
            #               torch.pow((1 + h_ * (1 + 2 * t[1])), n_))) / 3 +
            #           (grad_p1 / (torch.pow((1 + h_), n_)) + grad_t121 / (torch.pow((1 + h_), n_)) + grad_t122 / (
            #               torch.pow((1 + h_), n_))) / 3 +
            #           (grad_p1 / (torch.pow((1 + h_), n_)) + grad_t121 / (torch.pow((1 + h_), n_)) + grad_t122 / (
            #               torch.pow((1 + h_), n_))) / 3
            #           )/4
        if ctx.needs_input_grad[4]:
            grad_H1 = None
        if ctx.needs_input_grad[5]:
            grad_H2 = None
        if ctx.needs_input_grad[6]:
            grad_c = (grad_p1/ (torch.pow((1 + h_*(rho1)), n_))+grad_t121/ (torch.pow((1 + h_*(t[0]+t[1])+t[2]+t[3]), n_))+grad_t122/ (torch.pow((1 + h_*(t[0]+t[1])+t[2]+t[3]), n_)))/3

        return grad_rho1,grad_mu1,grad_v1,grad_t,grad_H1,grad_H2,grad_c


Z1_func = Z1_function.apply#为了使使用这些自定义操作变得更加容易，我们建议使用别名作为其 apply方法



class Z1_Layer(torch.nn.Module):
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
        super(Z1_Layer, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        # self.in_features = in_features
        # self.out_features = out_features

        #self.c = torch.nn.Parameter(torch.Tensor(1, 1))  # 由于weights是可以训练的，所以使用Parameter来定义

        # if bias:
        #     self.bias = torch.nn.Parameter(torch.Tensor(in_features))  # 由于bias是可以训练的，所以使用Parameter来定义
        # else:
        #     self.register_parameter('bias', None)

    def forward(self, rho1,mu1,v1,t,H1,H2,c):




        # rho1 = rho1.item()
        # mu1 = mu1.item()
        # v1 = matlab.double(v1.tolist())
        # t = matlab.double(t.tolist())
        # H1 = matlab.double(H1.tolist())
        # H2 = matlab.double(H2.tolist())
        # ret = mbegin.eng.Z1(rho1, mu1, v1, t, H1, H2, nargout=10)
        # W11 = torch.tensor(ret[0])
        # W12 =  torch.tensor(ret[1])
        # lambda111 =  torch.tensor(ret[2])
        # lambda112 =  torch.tensor(ret[3])
        # lambda121 =  torch.tensor(ret[4])
        # lambda122 =  torch.tensor(ret[5])
        # t121 =  torch.tensor(ret[6])
        # t122 =  torch.tensor(ret[7])
        # t211 =  torch.tensor(ret[8])
        # t212 =  torch.tensor(ret[9])
        # p1 = torch.trace(W11) + torch.trace(W12)
        p1, W11, W12, t121, t122, t211, t212 = Z1_func(rho1,mu1,v1,t,H1,H2,c)

        return p1,W11,W12,t121,t122,t211,t212
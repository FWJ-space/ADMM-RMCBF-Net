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





# 先定义一个模型
class ZXM_layer(torch.nn.Module):
    def __init__(self):
        super(ZXM_layer, self).__init__()  # 第一句话，调用父类的构造函数
        # Z_layer_net = torch.load('D:\Python36\Z_learn\Z_layer_net')

        self.Z1_layer = Z1_Layer()
        self.Z2_layer = Z2_Layer()
        self.Xt_layer = Xt_Layer()
        self.Xrho1_layer = Xrho1_Layer()
        self.Xrho2_layer = Xrho2_Layer()
        self.Mv1_layer = Mv1_Layer()
        self.Mv2_layer = Mv2_Layer()
        self.Mmu1_layer = Mmu1_Layer()
        self.Mmu2_layer = Mmu2_Layer()

        c = torch.tensor([5000.0], requires_grad=True)
        c = c.double()
        self.c = torch.nn.Parameter(c, requires_grad=True)  # 由于weights是可以训练的，所以使用Parameter来定义

    def forward(self,rho1,rho2,mu1,mu2,v1,v2,t,H1,H2,W11,W12,W21,W22,c):


       # 单一参数模式

       #  # Z_layer
       #  print("Z_layer")
       #  p1,W11,W12,t1211, t1221,t2111,t2121 = self.Z1_layer(rho1,mu1,v1,t,H1,H2,c)
       #  p2,W21,W22,t121, t122, t211, t212 = self.Z2_layer(rho2,mu2,v2,t,H1,H2,c)
       #  # X_layer
       #  print("X_layer")
       #  t1 = torch.Tensor([[t2111], [t2121], [t1211], [t1221]])
       #  t2 = torch.Tensor([[t121], [t122], [t211], [t212]])
       #  t=self.Xt_layer(t1,t2,v1,v2,c)
       #
       #
       #  rho1 = self.Xrho1_layer(p1,mu1,c)
       #  rho2 = self.Xrho2_layer(p2, mu2,c)
       #
       # # print("%.8f , %.8f" %(p11,p22))
       #  # M_layer
       #  print("M_layer")
       #  v1 = self.Mv1_layer(t,t1,v1,c)
       #  v2 = self.Mv2_layer(t, t2, v2,c)
       #  mu1 = self.Mmu1_layer(rho1 ,p1 , mu1,c)

       #  mu2 = self.Mmu2_layer(rho2, p2, mu2,c)



       #  每层一个参数模式

        # Z_layer
        print("Z_layer")
        p1,W11,W12,t1211, t1221,t2111,t2121 = self.Z1_layer(rho1,mu1,v1,t,H1,H2,self.c)
        p2,W21,W22,t121, t122, t211, t212 = self.Z2_layer(rho2,mu2,v2,t,H1,H2,self.c)
        # X_layer
        print("X_layer")
        t1 = torch.Tensor([[t2111], [t2121], [t1211], [t1221]])
        t2 = torch.Tensor([[t121], [t122], [t211], [t212]])
        t=self.Xt_layer(t1,t2,v1,v2,self.c)


        rho1 = self.Xrho1_layer(p1,mu1,self.c)
        rho2 = self.Xrho2_layer(p2, mu2,self.c)

       # print("%.8f , %.8f" %(p11,p22))
        # M_layer
        print("M_layer")
        v1 = self.Mv1_layer(t,t1,v1,self.c)
        v2 = self.Mv2_layer(t, t2, v2,self.c)
        mu1 = self.Mmu1_layer(rho1 ,p1 , mu1,self.c)
        mu2 = self.Mmu2_layer(rho2, p2, mu2,self.c)


        return rho1,rho2,mu1,mu2,v1,v2,t,H1,H2,W11,W12,W21,W22





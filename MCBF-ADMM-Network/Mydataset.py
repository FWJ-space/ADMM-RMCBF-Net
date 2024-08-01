from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch
import scipy.io as sio

class Mydataset(Dataset):
    def __init__(self):
        H_temp1 = sio.loadmat('D:\Python36/data_r.mat')
        H_temp2 = sio.loadmat('D:\Python36/data_i.mat')
        H1 = H_temp1["data_r"]
        H2 = H_temp2["data_i"]
        # H1 = H1.transpose((0,2,1))
        # H2 = H2.transpose((0,2,1))
        rho1 = torch.zeros(200,1)
        rho2 = torch.zeros(200,1)
        mu1 = torch.zeros(200,1)
        mu2 = torch.zeros(200,1)
        v1 = torch.zeros(200,4, 1)
        v2 = torch.zeros(200,4, 1)
        t = torch.zeros(200,4, 1)
        W11 = torch.zeros(200,8, 8)
        W12 = torch.zeros(200,8, 8)
        W21 = torch.zeros(200,8, 8)
        W22 = torch.zeros(200,8, 8)

        P_out_temp = sio.loadmat('D:\Python36/P_out.mat')
        P_out = P_out_temp["P_out"]

        P1_out = P_out[0]
        P2_out = P_out[1]

        self.H1=torch.from_numpy(H1).double()
        self.H2 = torch.from_numpy(H2).double()
        self.rho1 = rho1.double()
        self.rho2 = rho2.double()
        self.mu1 = mu1.double()
        self.mu2 = mu2.double()
        self.v1 = v1.double()
        self.v2 = v2.double()
        self.t = t.double()
        self.W11 = W11.double()
        self.W12 = W12.double()
        self.W21 = W21.double()
        self.W22 = W22.double()
        self.P1_out = torch.from_numpy(P1_out).double()
        self.P2_out = torch.from_numpy(P2_out).double()
        self.len = H1.shape[0]
    # stuff

    def __getitem__(self, index):
        # stuff
        lplus = 50
        return self.rho1[index],self.rho2[index],self.mu1[index],self.mu2[index],self.v1[index],self.v2[index],self.t[index],self.H1[index+lplus],self.H2[index+lplus],self.W11[index],self.W12[index],self.W21[index],self.W22[index],self.P1_out[index+lplus],self.P2_out[index+lplus]

    def __len__(self):

        return 20


class Mydataset_test(Dataset):
    def __init__(self):
        H_temp1 = sio.loadmat('D:\Python36/data_r.mat')
        H_temp2 = sio.loadmat('D:\Python36/data_i.mat')
        H1 = H_temp1["data_r"]
        H2 = H_temp2["data_i"]
        # H1 = H1.transpose((0,2,1))
        # H2 = H2.transpose((0,2,1))
        rho1 = torch.zeros(200,1)
        rho2 = torch.zeros(200,1)
        mu1 = torch.zeros(200,1)
        mu2 = torch.zeros(200,1)
        v1 = torch.zeros(200,4, 1)
        v2 = torch.zeros(200,4, 1)
        t = torch.zeros(200,4, 1)
        W11 = torch.zeros(200,8, 8)
        W12 = torch.zeros(200,8, 8)
        W21 = torch.zeros(200,8, 8)
        W22 = torch.zeros(200,8, 8)

        P_out_temp = sio.loadmat('D:\Python36/P_out.mat')
        P_out = P_out_temp["P_out"]

        P1_out = P_out[0]
        P2_out = P_out[1]

        self.H1=torch.from_numpy(H1).double()
        self.H2 = torch.from_numpy(H2).double()
        self.rho1 = rho1.double()
        self.rho2 = rho2.double()
        self.mu1 = mu1.double()
        self.mu2 = mu2.double()
        self.v1 = v1.double()
        self.v2 = v2.double()
        self.t = t.double()
        self.W11 = W11.double()
        self.W12 = W12.double()
        self.W21 = W21.double()
        self.W22 = W22.double()
        self.P1_out = torch.from_numpy(P1_out).double()
        self.P2_out = torch.from_numpy(P2_out).double()
        self.len = H1.shape[0]
    # stuff

    def __getitem__(self, index):
        # stuff
        lplus = 50
        return self.rho1[index], self.rho2[index], self.mu1[index], self.mu2[index], self.v1[index], self.v2[index], \
               self.t[index], self.H1[index + lplus], self.H2[index + lplus], self.W11[index], self.W12[index], \
               self.W21[index], self.W22[index], self.P1_out[index + lplus], self.P2_out[index + lplus]

    def __len__(self):

        return 40
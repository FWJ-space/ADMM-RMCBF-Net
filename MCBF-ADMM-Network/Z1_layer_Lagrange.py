import torch
import matlab.engine
import mbegin
import scipy.io as sio





class Z1_Layer_start(torch.nn.Module):

    def __init__(self):
        super(Z1_Layer_start, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数

    def forward(self, rho1,mu1,v1,t,H1,c):
        # 参数定义


        gamma = torch.tensor(10.).double()
        gamma1 = (torch.div(1.,gamma))
        epsilon = torch.tensor(0.05).double()
        Q=(torch.div(1.,epsilon))*(torch.div(1.,epsilon))
        sigma = torch.tensor(0.02).double()
        sigma2 = sigma*sigma
        # 信道向量
        H1=0.2*H1
        h111 = H1[:, 0]
        h112 = H1[:, 2]
        h221 = H1[:, 4]
        h222 = H1[:, 6]
        h121 = H1[:, 5]
        h122 = H1[:, 7]
        h211 = H1[:, 1]
        h212 = H1[:, 3]
        # LMI不等式系数矩阵
        m=70 # 条件矩阵维度
        n=85 # 自变量个数
        # 常数矩阵
        A0=torch.zeros([m,m],dtype=torch.double)
        # 常数块
        A0[8, 8] = -sigma2
        A0[17, 17] = -sigma2
        A0[61, 61] = 1.
        A0[63, 63] = 1.
        A0[65, 65] = 1.
        A0[67, 67] = 1.
        A0[69, 69] = 1.
        # t 3412
        A0[60, 61] = t[2,0]
        A0[61, 60] = t[2,0]
        A0[62, 63] = t[3,0]
        A0[63, 62] = t[3,0]
        A0[64, 65] = t[0,0]
        A0[65, 64] = t[0,0]
        A0[66, 67] = t[1,0]
        A0[67, 66] = t[1,0]
        # W11

        A = torch.zeros([8,8,m,m],dtype=torch.double)

        for i in range(8):
            for j in range(8):
                # 1
                # A[k,:,:]=torch.zeros(m)
                A[i, j, i, j] = gamma1
                A[i, j, i, 8] = gamma1 * h111[j]
                A[i, j, 8, j] = gamma1 * h111[i]
                A[i, j, 8, 8] = gamma1 * h111[i] * h111[j]
                # 2
                A[i, j, i+9, j+9] = -1.
                A[i, j, i + 9, 17] = -h112[j]
                A[i, j, 17, j+9] = -h112[i]
                A[i, j, 17, 17] = -h112[i]*h112[j]
                # 3
                A[i, j, i + 18, j + 18] = -1.
                A[i, j, i + 18, 26] = -h121[j]
                A[i, j, 26, j + 18] = -h121[i]
                A[i, j, 26, 26] = -h121[i]*h121[j]
                # 4
                A[i, j, i + 27, j + 27] = -1.
                A[i, j, i + 27, 35] = -h122[j]
                A[i, j, 35, j + 27] = -h122[i]
                A[i, j, 35, 35] = -h122[i] * h122[j]
                # 5
                A[i,j,i+36,j+36]=1
                # 8*8 5*5 8*8
            A[i,i,68,69]=1
            A[i, i, 69, 68] = 1

        Asvec=torch.zeros([36,m,m],dtype=torch.double)
        k=0
        for i in range(8):
            for j in range(i,8):
                if i==j:
                    Asvec[k,:,:]=A[i,j,:,:]
                else:
                    Asvec[k, :, :] = A[i, j, :, :]+A[j, i, :, :]
                k=k+1


        A1 = Asvec[1, :, :].numpy()
        A2 = Asvec[2, :, :].numpy()
        A3 = Asvec[3, :, :].numpy()


        # W12

        B = torch.zeros([8, 8, m, m], dtype=torch.double)

        for i in range(8):
            for j in range(8):
                # 1
                # A[k,:,:]=torch.zeros(m)
                B[i, j, i, j] = -1.
                B[i, j, i, 8] = -h111[j]
                B[i, j, 8, j] = -h111[i]
                B[i, j, 8, 8] = -h111[i] * h111[j]
                # 2
                B[i, j, i + 9, j + 9] = gamma1
                B[i, j, i + 9, 17] = gamma1*h112[j]
                B[i, j, 17, j + 9] = gamma1*h112[i]
                B[i, j, 17, 17] = gamma1*h112[i] * h112[j]
                # 3
                B[i, j, i + 18, j + 18] = -1.
                B[i, j, i + 18, 26] = -h121[j]
                B[i, j, 26, j + 18] = -h121[i]
                B[i, j, 26, 26] = -h121[i] * h121[j]
                # 4
                B[i, j, i + 27, j + 27] = -1.
                B[i, j, i + 27, 35] = -h122[j]
                B[i, j, 35, j + 27] = -h122[i]
                B[i, j, 35, 35] = -h122[i] * h122[j]
                # 5
                B[i, j, i + 44, j + 44] = 1
                # 8*8 5*5 8*8
            B[i, i, 68, 69] = 1
            B[i, i, 69, 68] = 1




        Bsvec=torch.zeros([36,m,m],dtype=torch.double)
        k=0
        for i in range(8):
            for j in range(i,8):
                if i==j:
                    Bsvec[k,:,:]=B[i,j,:,:]
                else:
                    Bsvec[k, :, :] = B[i, j, :, :]+B[j, i, :, :]
                k=k+1

        B1 = Bsvec[1, :, :].numpy()
        B2 = Bsvec[2, :, :].numpy()
        B3 = Bsvec[3, :, :].numpy()


        A2 = torch.zeros([13, m, m], dtype=torch.double)
        #lambda111
        A2[0,0:8,0:8]=Q*torch.eye(8)
        A2[0,8,8]=-1.
        A2[0,52,52]=1.
        #lambda112
        A2[1,9:17,9:17]=Q*torch.eye(8)
        A2[1,17,17]=-1.
        A2[1,53,53]=1.
        #lambda121
        A2[2,18:26,18:26]=Q*torch.eye(8)
        A2[2,26,26]=-1.
        A2[2,54,54]=1.
        #lambda122
        A2[3,27:35,27:35]=Q*torch.eye(8)
        A2[3,35,35]=-1.
        A2[3,55,55]=1.

        #t211
        A2[4,8,8]= -1.
        A2[4,56,56]=1.
        A2[4,60,61]=1.
        A2[4, 61, 60] = 1.
        #t212
        A2[5,17,17]=-1.
        A2[5,57,57]=1.
        A2[5,62,63]=1.
        A2[5,63,62]=1.
        #t121
        A2[6, 26, 26] = 1.
        A2[6, 58, 58] = 1.
        A2[6, 64, 65] = 1.
        A2[6, 65, 64] = 1.
        #t122
        A2[7, 35, 35] = 1.
        A2[7, 59, 59] = 1.
        A2[7, 66, 67] = 1.
        A2[7, 67, 66] = 1.
        #t0211
        A2[8, 60, 60] = 1.
        #t0212
        A2[9, 62, 62] = 1.
        #t0121
        A2[10, 64, 64] = 1.
        #t0122
        A2[11, 66, 66] = 1.
        #p0
        A2[12, 68, 68] = 1.

        b = torch.zeros([n, 1], dtype=torch.double)
        #W11
        b[0, 0] = 1.
        b[8, 0] = 1.
        b[15, 0] = 1.
        b[21, 0] = 1.
        b[26, 0] = 1.
        b[30, 0] = 1.
        b[33, 0] = 1.
        b[35, 0] = 1.
        #W12
        b[36+0, 0] = 1.
        b[36 + 8, 0] = 1.
        b[36 + 15, 0] = 1.
        b[36 + 21, 0] = 1.
        b[36 + 26, 0] = 1.
        b[36 + 30, 0] = 1.
        b[36 + 33, 0] = 1.
        b[36 + 35, 0] = 1.
        #t211 t212 t121 t122
        b[76,0] = v1[0,0]
        b[77, 0] = v1[1,0]
        b[78, 0] = v1[2,0]
        b[79, 0] = v1[3,0]

        #p0
        b[84,0]=c/2
        #t0211 t0212 t0121 t122
        b[80,0]=c/2
        b[81, 0] = c / 2
        b[82, 0] = c / 2
        b[83, 0] = c / 2


        AA = torch.zeros([n, m,m], dtype=torch.double)

        AA[0:36,:,:]=Asvec
        AA[36:72,:,:]=Bsvec

        for i in range(13):
            AA[72+i,:,:]=A2[i,:,:]


        AA=-AA
        b=-b


        return AA,b,A0


class Z2_Layer_start(torch.nn.Module):

    def __init__(self):
        super(Z2_Layer_start, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数

    def forward(self, rho1,mu1,v1,t,H1,c):
        # 参数定义


        gamma = torch.tensor(10.).double()
        gamma1 = (torch.div(1.,gamma))
        epsilon = torch.tensor(0.05).double()
        Q=(torch.div(1.,epsilon))*(torch.div(1.,epsilon))
        sigma = torch.tensor(0.02).double()
        sigma2 = sigma*sigma
        # 信道向量
        H1=0.2*H1
        h111 = H1[:, 0]
        h112 = H1[:, 2]
        h221 = H1[:, 4]
        h222 = H1[:, 6]
        h121 = H1[:, 5]
        h122 = H1[:, 7]
        h211 = H1[:, 1]
        h212 = H1[:, 3]
        # LMI不等式系数矩阵
        m=70 # 条件矩阵维度
        n=85 # 自变量个数
        # 常数矩阵
        A0=torch.zeros([m,m],dtype=torch.double)
        # 常数块
        A0[8, 8] = -sigma2
        A0[17, 17] = -sigma2
        A0[61, 61] = 1.
        A0[63, 63] = 1.
        A0[65, 65] = 1.
        A0[67, 67] = 1.
        A0[69, 69] = 1.
        # t 3412
        A0[60, 61] = t[0,0]
        A0[61, 60] = t[0,0]
        A0[62, 63] = t[1,0]
        A0[63, 62] = t[1,0]
        A0[64, 65] = t[2,0]
        A0[65, 64] = t[2,0]
        A0[66, 67] = t[3,0]
        A0[67, 66] = t[3,0]
        # W11

        A = torch.zeros([8,8,m,m],dtype=torch.double)

        for i in range(8):
            for j in range(8):
                # 1
                # A[k,:,:]=torch.zeros(m)
                A[i, j, i, j] = gamma1
                A[i, j, i, 8] = gamma1 * h221[j]
                A[i, j, 8, j] = gamma1 * h221[i]
                A[i, j, 8, 8] = gamma1 * h221[i] * h221[j]
                # 2
                A[i, j, i+9, j+9] = -1.
                A[i, j, i + 9, 17] = -h222[j]
                A[i, j, 17, j+9] = -h222[i]
                A[i, j, 17, 17] = -h222[i]*h222[j]
                # 3
                A[i, j, i + 18, j + 18] = -1.
                A[i, j, i + 18, 26] = -h211[j]
                A[i, j, 26, j + 18] = -h211[i]
                A[i, j, 26, 26] = -h211[i]*h211[j]
                # 4
                A[i, j, i + 27, j + 27] = -1.
                A[i, j, i + 27, 35] = -h212[j]
                A[i, j, 35, j + 27] = -h212[i]
                A[i, j, 35, 35] = -h212[i] * h212[j]
                # 5
                A[i,j,i+36,j+36]=1
                # 8*8 5*5 8*8
            A[i,i,68,69]=1
            A[i, i, 69, 68] = 1

        Asvec=torch.zeros([36,m,m],dtype=torch.double)
        k=0
        for i in range(8):
            for j in range(i,8):
                if i==j:
                    Asvec[k,:,:]=A[i,j,:,:]
                else:
                    Asvec[k, :, :] = A[i, j, :, :]+A[j, i, :, :]
                k=k+1





        # W12

        B = torch.zeros([8, 8, m, m], dtype=torch.double)

        for i in range(8):
            for j in range(8):
                # 1
                # A[k,:,:]=torch.zeros(m)
                B[i, j, i, j] = -1.
                B[i, j, i, 8] = -h221[j]
                B[i, j, 8, j] = -h221[i]
                B[i, j, 8, 8] = -h221[i] * h221[j]
                # 2
                B[i, j, i + 9, j + 9] = gamma1
                B[i, j, i + 9, 17] = gamma1*h222[j]
                B[i, j, 17, j + 9] = gamma1*h222[i]
                B[i, j, 17, 17] = gamma1*h222[i] * h222[j]
                # 3
                B[i, j, i + 18, j + 18] = -1.
                B[i, j, i + 18, 26] = -h211[j]
                B[i, j, 26, j + 18] = -h211[i]
                B[i, j, 26, 26] = -h211[i] * h211[j]
                # 4
                B[i, j, i + 27, j + 27] = -1.
                B[i, j, i + 27, 35] = -h212[j]
                B[i, j, 35, j + 27] = -h212[i]
                B[i, j, 35, 35] = -h212[i] * h212[j]
                # 5
                B[i, j, i + 44, j + 44] = 1
                # 8*8 5*5 8*8
            B[i, i, 68, 69] = 1
            B[i, i, 69, 68] = 1




        Bsvec=torch.zeros([36,m,m],dtype=torch.double)
        k=0
        for i in range(8):
            for j in range(i,8):
                if i==j:
                    Bsvec[k,:,:]=B[i,j,:,:]
                else:
                    Bsvec[k, :, :] = B[i, j, :, :]+B[j, i, :, :]
                k=k+1



        A2 = torch.zeros([13, m, m], dtype=torch.double)
        #lambda111
        A2[0,0:8,0:8]=Q*torch.eye(8)
        A2[0,8,8]=-1.
        A2[0,52,52]=1.
        #lambda112
        A2[1,9:17,9:17]=Q*torch.eye(8)
        A2[1,17,17]=-1.
        A2[1,53,53]=1.
        #lambda121
        A2[2,18:26,18:26]=Q*torch.eye(8)
        A2[2,26,26]=-1.
        A2[2,54,54]=1.
        #lambda122
        A2[3,27:35,27:35]=Q*torch.eye(8)
        A2[3,35,35]=-1.
        A2[3,55,55]=1.

        #t121
        A2[4,8,8]= -1.
        A2[4,56,56]=1.
        A2[4,60,61]=1.
        A2[4, 61, 60] = 1.
        #t122
        A2[5,17,17]=-1.
        A2[5,57,57]=1.
        A2[5,62,63]=1.
        A2[5,63,62]=1.
        #t212
        A2[6, 26, 26] = 1.
        A2[6, 58, 58] = 1.
        A2[6, 64, 65] = 1.
        A2[6, 65, 64] = 1.
        #t212
        A2[7, 35, 35] = 1.
        A2[7, 59, 59] = 1.
        A2[7, 66, 67] = 1.
        A2[7, 67, 66] = 1.
        #t0121
        A2[8, 60, 60] = 1.
        #t0122
        A2[9, 62, 62] = 1.
        #t0211
        A2[10, 64, 64] = 1.
        #t0212
        A2[11, 66, 66] = 1.
        #p0
        A2[12, 68, 68] = 1.

        b = torch.zeros([n, 1], dtype=torch.double)
        #W11
        b[0, 0] = 1.
        b[8, 0] = 1.
        b[15, 0] = 1.
        b[21, 0] = 1.
        b[26, 0] = 1.
        b[30, 0] = 1.
        b[33, 0] = 1.
        b[35, 0] = 1.
        #W12
        b[36+0, 0] = 1.
        b[36 + 8, 0] = 1.
        b[36 + 15, 0] = 1.
        b[36 + 21, 0] = 1.
        b[36 + 26, 0] = 1.
        b[36 + 30, 0] = 1.
        b[36 + 33, 0] = 1.
        b[36 + 35, 0] = 1.
        #t211 t212 t121 t122
        b[76,0] = v1[2,0]
        b[77, 0] = v1[3,0]
        b[78, 0] = v1[0,0]
        b[79, 0] = v1[1,0]

        #p0
        b[84,0]=c/2
        #t0211 t0212 t0121 t122
        b[80,0]=c/2
        b[81, 0] = c / 2
        b[82, 0] = c / 2
        b[83, 0] = c / 2


        AA = torch.zeros([n, m,m], dtype=torch.double)

        AA[0:36,:,:]=Asvec
        AA[36:72,:,:]=Bsvec

        for i in range(13):
            AA[72+i,:,:]=A2[i,:,:]


        AA=-AA
        b=-b


        return AA,b,A0



class IPM1(torch.nn.Module):

    def __init__(self):
        super(IPM1, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        self.svec = svec()

    def forward(self, A1,b,c):


        #parameters
        one=torch.tensor(1.).double()
        # m = number of constraints
        m=85
        # n = sum( n's )
        n=70
        n1=torch.tensor(70.).double()
        n2=4900
        nt=2485
        # convert A and c to svec versions
        c=self.svec(c)
        A_temp = A1
        A = torch.zeros([nt,m],dtype=torch.double)
        Ai = torch.zeros([nt, 1], dtype=torch.double)
        for i in range(m):
            Ai= self.svec(A_temp[i,:,:])
            A=torch.cat((A,Ai),1)
        A=A[:,85:170]
        A=A.T

        # A=A[0:35,:]

        # set ksi
        ksi=torch.tensor(0).double()
        temp = torch.zeros([m,1],dtype=torch.double)
        for k in range(m):
            temp[k,0]=(1+torch.abs(b[k,0])) /(1+ torch.norm(A[k,:]))

        ksi = n*torch.max(temp)

        #set eta
        eta = torch.tensor(0).double()
        temp = torch.zeros([m, 1], dtype=torch.double)
        for k in range(m):
            temp[k,0]=torch.norm(A[k,:])

        eta = (torch.div(one,torch.sqrt(n1)))*(1.+torch.max(torch.max(temp),torch.norm(c)))
        # set initial iterates using ksi and eta
        X = ksi*torch.eye(n).double()
        x = self.svec(X)
        y = torch.zeros([m,1]).double()
        Z = eta*torch.eye(n).double()
        z = self.svec(Z)

        #calculate normA for each block, save sqrt of each

        normsA = torch.norm(A,p="fro")
        normsA = torch.max(one,torch.sqrt(normsA))

        #calculate normb
        normb = torch.max(one,torch.norm(b))


        #calculate normc for each block, save max
        normc = torch.max(one,torch.norm(c))

        #scale constraints
        A = torch.mul(A,torch.div(one,normsA))
        c = torch.mul(c,torch.div(one,normc*normsA))
        b = torch.mul(b,torch.div(one,normb))

        # scale initial iterates x and z
        x = torch.mul(x, normsA)
        z = torch.mul(z,torch.div(one,normc*normsA))

        #set initial values
        rp = b - torch.matmul(A,x)
        Rd = c- z - torch.matmul(A.T,y)
        # relgap = torch.div(torch.mul(x,z),(one+torch.max(torch.abs(torch.mul(c,x)) ,
        #                                                  torch.abs(torch.mul(b,y)) )))
        pinfeas = torch.div(torch.norm(rp),(one+torch.norm(b)))
        dinfeas = torch.div(torch.norm(Rd),(one+torch.norm(c)))
        phi = torch.max(pinfeas,dinfeas)
        soln_relgap = torch.tensor(0.000001).double()
        soln_phi = torch.tensor(0.000001).double()
        g = torch.tensor(0.9).double()
        iter = 0


        return x,y,z,A,b,c,normc



class IPM2(torch.nn.Module):

    def __init__(self):
        super(IPM2, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        self.svec = svec()
        self.smat = smat()
        self.Hp = Hp()
        self.skmult = skmult()


    def forward(self, x,y,z,A,b,c,normc):
        # parameters
        one = torch.tensor(1.).double()
        g = torch.tensor(0.9).double()
        soln_relgap = torch.tensor(0.000001).double()

        n = 70
        n1 = torch.tensor(70.).double()
        nt = 2485
        #get Cholesky decompositions of X and Z (upper triangular!)
        X = self.smat(x)
        Q = torch.cholesky(X,upper=True)
        Z = self.smat(z)
        P = torch.cholesky(Z,upper=True)
        #calculate residuals, relgap, feasability
        rp = b - torch.matmul(A, x)
        Rd = c - z - torch.matmul(A.T, y)

        mu = torch.div(torch.matmul(x.T[0],z.T[0]),n1)
        # relgap = torch.div(torch.mul(x, z), (one + torch.max(torch.abs(torch.mul(c, x)), torch.abs(torch.mul(b, y)))))
        pinfeas = torch.div(torch.norm(rp),(one+torch.norm(b)))
        dinfeas = torch.div(torch.norm(Rd),(one+torch.norm(c)))
        phi = torch.max(pinfeas,dinfeas)
        Rc = -self.svec(self.Hp(torch.matmul(X,Z),P))


        rpnum = rp.numpy()
        Rdnum = Rd.numpy()
        Rcnum = Rc.numpy()
        Pnum =P.numpy()

        # solve system (predictor)
        XZA = self.skmult(X,torch.inverse(Z),A.T).numpy()
        M = torch.mm(A,self.skmult(X,torch.inverse(Z),A.T))
        h = rp + torch.matmul(A,(self.skmult(X,torch.inverse(Z),Rd) -
                                 self.skmult(torch.inverse(P),torch.inverse(P),Rc)
                                 ) )

        Mn = M.numpy()

        Minv=torch.inverse(M).numpy()
        hum=h.numpy()
        # dy2 = torch.matmul(h, torch.inverse(M))

        dy = torch.mm(torch.inverse(M) , h)

        dz = Rd -torch.matmul(A.T,dy)
        dx = -x -self.skmult(X,torch.inverse(Z),dz)

        #calculate alpha (primal step-length)

        alphas = torch.tensor(0.0,dtype= torch.double )
        XDX = torch.matmul(torch.matmul(torch.inverse(Q.T) , self.smat(dx)) , torch.inverse(Q))
        eigs_a = torch.symeig(XDX,eigenvectors=False)
        lambda_a = eigs_a[0]
        lambda_a = lambda_a[0]
        if lambda_a <0 :
            alphas = -torch.div(one,lambda_a)
        else:
            alphas = 20*one

        alpha = torch.min(0.5*one,g*alphas)

        betas = torch.tensor(0.0,dtype= torch.double )
        ZDZ = torch.matmul(torch.matmul(torch.inverse(P.T) , self.smat(dz)) , torch.inverse(P))
        eigs_b = torch.symeig(ZDZ, eigenvectors=False)
        lambda_b = eigs_b[0]
        lambda_b = lambda_b[0]
        if lambda_b < 0:
            betas = -torch.div(one,lambda_b)
        else:
            betas = 20*one

        beta = torch.min(0.5*one,g*betas)
        # calculate exponent (for sigma)
        if mu > soln_relgap:
            if torch.min(alpha,beta) < torch.sqrt(torch.div(one,torch.tensor(3.,dtype=torch.double))) :
                e=1
            else:
                e = torch.max(one,3*one*torch.min(alpha,beta)*torch.min(alpha,beta))
        else:
            e = one

        #calculate sigma (centering parameter)
        if torch.matmul((x+alpha*dx).T[0],(z+beta*dz).T[0])<0:
            sigma = 0.8*one
        else:
            frac = torch.div(torch.matmul((x+alpha*dx).T[0],(z+beta*dz).T[0]) , torch.matmul(x.T[0],z.T[0]))
            sigma = torch.min(one,torch.pow(frac,e))

        #solve modified system (corrector)
        XZ = torch.matmul(X, Z)

        HXZ=self.Hp(torch.matmul(X, Z), P)
        SME = sigma*mu*torch.eye(n)
        SDX=self.smat(dx)
        SDY=self.smat(dz)
        DxDy =torch.matmul(self.smat(dx),self.smat(dz))
        HDxDy = self.Hp(torch.matmul(self.smat(dx),self.smat(dz)),P)
        Rc = self.svec(sigma*mu*torch.eye(n)-self.Hp(torch.matmul(X,Z),P)-self.Hp(torch.matmul(self.smat(dx),self.smat(dz)),P))
        h = rp + torch.matmul(A, (self.skmult(X, torch.inverse(Z), Rd) -
                                  self.skmult(torch.inverse(P), torch.inverse(P), Rc)
                                  ))

        dy = torch.matmul(torch.inverse(M) , h)
        dz = Rd -torch.matmul(A.T,dy)
        dx = self.skmult(torch.inverse(P), torch.inverse(P), Rc) - self.skmult(X, torch.inverse(Z), dz)
        g = 0.9*one +0.09*torch.min(alpha,beta)
        # calculate alpha (primal step-length)

        alphas = torch.tensor(0.0, dtype=torch.double)
        XDX = torch.matmul(torch.matmul(torch.inverse(Q.T), self.smat(dx)), torch.inverse(Q))
        eigs_a = torch.symeig(XDX, eigenvectors=False)
        lambda_a = eigs_a[0]
        lambda_a = lambda_a[0]
        if lambda_a < 0:
            alphas = -torch.div(one, lambda_a)
        else:
            alphas = 20 * one

        alpha = torch.min(0.5*one, g*alphas)

        betas = torch.tensor(0.0, dtype=torch.double)
        ZDZ = torch.matmul(torch.matmul(torch.inverse(P.T), self.smat(dz)), torch.inverse(P))
        eigs_b = torch.symeig(ZDZ, eigenvectors=False)
        lambda_b = eigs_b[0]
        lambda_b = lambda_b[0]
        if lambda_b < 0:
            betas = -torch.div(one, lambda_b)
        else:
            betas = 20 * one

        beta = torch.min(0.5*one, g*betas)
        #update solution
        x = x + alpha * dx
        y = y + beta * dy
        z = z + beta * dz
        g = 0.9 * one + 0.09 * torch.min(alpha, beta)
        primal = torch.matmul(c.T[0],x.T[0])
        dual = torch.matmul(b.T[0],y.T[0])
        pinfeas = torch.div(torch.norm(rp),(one+torch.norm(b)))
        dinfeas = torch.div(torch.norm(Rd),(one+torch.norm(c)))

        y1=y*normc
        print("原始结果 %.10f  " % (primal))


        return x,y,z,A,b,c,normc,y1




class svec(torch.nn.Module):

    def __init__(self):
        super(svec, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数


    def forward(self, X):
        n = 70
        nt = 2485

        # n=8
        # nt=36

        # x=torch.zeros([nt,1],dtype=torch.double)
        # for i in range(1,n+1):
        #     x[int((i*(i+1))/2-1),0]=X[(i-1),(i-1)]
        #
        # start=2
        # for i in range(2,n+1):
        #     x[(start-1):(start+i-2),0]=X[0:int(i-1),(i-1)]*torch.sqrt(torch.tensor(2.0).double())
        #     start=start+i

        X=torch.sqrt(torch.tensor(2.0).double())*X-(torch.sqrt(torch.tensor(2.0).double())-torch.tensor(1.0).double())*X*torch.eye(n)
        select = torch.zeros((n,n),dtype=torch.bool)
        for i in range(n):
            select[i,:i+1] = 1

        x = X[select]
        x=torch.reshape(x,(nt,1))

        return x




class smat(torch.nn.Module):

    def __init__(self):
        super(smat, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数


    def forward(self, x):
        #将对称块向量转换为块对角矩阵
        n = 70
        nt = 2485
        # n=8
        # nt=36
        X=torch.zeros([n,n]).double()

        x1=x.reshape(-1)



        select = torch.zeros((n,n),dtype=torch.bool)
        for i in range(n):
            select[i,:i+1] = 1

        X[select]=x1

        X=torch.sqrt(torch.tensor(0.5).double())*(X+X.T)-(torch.sqrt(torch.tensor(2.0).double())-torch.tensor(1.0).double())*X*torch.eye(n)

        # #复制右上角
        # start = 1
        # for i in range(1,(n+1)):
        #     ii=i-1
        #     X[0:i,ii] = x[start-1:start+i-1,0]
        #     start=start+i
        #
        # #填写左下角 除根号2
        # for i in range(1,n+1):
        #     for j in range(i+1,n+1):
        #         X[i-1,j-1] = X[i-1,j-1]*torch.sqrt(torch.tensor(0.5).double())
        #         X[j - 1, i - 1] = X[i - 1, j - 1]





        return X

class smat2(torch.nn.Module):

    def __init__(self):
        super(smat2, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数


    def forward(self, x):
        #将对称块向量转换为块对角矩阵
        n = 70
        nt = 2485
        # n=8
        # nt=36
        # X=torch.zeros([n,n])
        #
        # #复制右上角
        # start = 1
        # for i in range(1,(n+1)):
        #     ii=i-1
        #     X[0:i,ii] = x[start-1:start+i-1]
        #     start=start+i
        #
        # #填写左下角 除根号2
        # for i in range(1,n+1):
        #     for j in range(i+1,n+1):
        #         X[i-1,j-1] = X[i-1,j-1]*torch.sqrt(torch.tensor(0.5).double())
        #         X[j - 1, i - 1] = X[i - 1, j - 1]
        X=torch.zeros([n,n]).double()

        x1=x



        select = torch.zeros((n,n),dtype=torch.bool)
        for i in range(n):
            select[i,:i+1] = 1

        X[select]=x1

        X=torch.sqrt(torch.tensor(0.5).double())*(X+X.T)-(torch.sqrt(torch.tensor(2.0).double())-torch.tensor(1.0).double())*X*torch.eye(n)

        return X



class Hp(torch.nn.Module):

    def __init__(self):
        super(Hp, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数



    def forward(self, U,P):
        # n = 70
        # nt = 2485
        two1 = torch.tensor(0.5).double()
        X = two1 * (torch.matmul(torch.matmul(P,U), torch.inverse(P) ) +
                    torch.matmul(torch.inverse(P.T), torch.matmul(U.T, P.T))
                    )






        return X

class skmult(torch.nn.Module):

    def __init__(self):
        super(skmult, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
        self.svec = svec()
        self.smat2 = smat2()


    def forward(self, G,K,A):
        # n = 70
        # nt = 2485
        two1 = torch.tensor(0.5).double()
        M = A ##初始化
        len =A.shape[1]



        for k in range(len):

            AG=two1*self.svec( torch.matmul(torch.matmul(K,self.smat2(A[:,k])), G.T ) +
                                        torch.matmul(
                                            torch.matmul(
                                                G,
                                                self.smat2(A[:,k])),
                                            K.T)

            )
            # M[:,k]= two1*self.svec( torch.matmul(torch.matmul(K,self.smat2(A[:,k])), G.T ) +
            #                             torch.matmul(
            #                                 torch.matmul(
            #                                     G,
            #                                     self.smat2(A[:,k])),
            #                                 K.T)
            #
            # )
            M=torch.cat((M,AG),1)
        M=M[:,len:2*len]








        return M

# class Z1_Layer_Lagrange(torch.nn.Module):
#     '''
#     因为这个层实现的功能是：
#      % M层
#         v1=v1+lamda*(E1*t-t1);
#         mu1=mu1+lamda*(rho1-p1);
#         v2=v2+lamda*(E2*t-t2);
#         mu2=mu2+lamda*(rho2-p2);
#     所以有一个个参数：lamda(lambda被占用)
#
#     输入 t,t1 v1的维度是（4*1)
#     输出 v1 的维度是（4*1)
#     '''
#
#     def __init__(self):
#         super(Z1_Layer_Lagrange, self).__init__()  # 和自定义模型一样，第一句话就是调用父类的构造函数
#         # self.in_features = in_features
#         # self.out_features = out_features
#
#         # self.c = torch.nn.Parameter(torch.Tensor(1, 1))  # 由于weights是可以训练的，所以使用Parameter来定义
#
#         # if bias:
#         #     self.bias = torch.nn.Parameter(torch.Tensor(in_features))  # 由于bias是可以训练的，所以使用Parameter来定义
#         # else:
#         #     self.register_parameter('bias', None)
#
#     def forward(self, rho1, mu1, v1, t, H1, H2, c):
#         # rho1 = rho1.item()
#         # mu1 = mu1.item()
#         # v1 = matlab.double(v1.tolist())
#         # t = matlab.double(t.tolist())
#         # H1 = matlab.double(H1.tolist())
#         # H2 = matlab.double(H2.tolist())
#         # ret = mbegin.eng.Z1(rho1, mu1, v1, t, H1, H2, nargout=10)
#         # W11 = torch.tensor(ret[0])
#         # W12 =  torch.tensor(ret[1])
#         # lambda111 =  torch.tensor(ret[2])
#         # lambda112 =  torch.tensor(ret[3])
#         # lambda121 =  torch.tensor(ret[4])
#         # lambda122 =  torch.tensor(ret[5])
#         # t121 =  torch.tensor(ret[6])
#         # t122 =  torch.tensor(ret[7])
#         # t211 =  torch.tensor(ret[8])
#         # t212 =  torch.tensor(ret[9])
#         # p1 = torch.trace(W11) + torch.trace(W12)
#
#         return p1, W11, W12, t121, t122, t211, t212



# 先定义一个模型
class Z1SDPSlover(torch.nn.Module):
    def __init__(self):
        super(Z1SDPSlover, self).__init__()  # 第一句话，调用父类的构造函数
        self.Z1_Layer_start = Z1_Layer_start()
        self.IPM1 = IPM1()
        self.IPM2 = IPM2()

    def forward(self, rho1,mu1,v1,t,H1,H2,c):

        AA,b,A0 = self.Z1_Layer_start(rho1,mu1,v1,t,H1,c)
        x, y, z, A, b, c, normc = self.IPM1(AA,b,A0)
        for stage in range(100):
            x, y, z, A, b, c, normc, y1 = self.IPM2(x, y, z, A, b, c, normc)
            print('stage: {}'.format(stage + 1))

        x, y, z, A, b, c, normc, y1 = self.IPM2(x, y, z, A, b, c, normc)

        p1 = y1[0, 0] + y1[8, 0] + y1[15, 0] + y1[21, 0] + y1[26, 0] + y1[30, 0] + y1[33, 0] + y1[35, 0] + y1[
            36 + 0, 0] + y1[36 + 8, 0] + y1[36 + 15, 0] + y1[36 + 21, 0] + y1[36 + 26, 0] + y1[36 + 30, 0] + y1[
                 36 + 33, 0] + y1[36 + 35, 0]
        t121=y1[78,0]
        t122=y1[79,0]
        t211=y1[76,0]
        t212=y1[77,0]



        return p1,t121, t122, t211, t212


class Z1SDPSloveri(torch.nn.Module):
    def __init__(self):
        super(Z1SDPSloveri, self).__init__()  # 第一句话，调用父类的构造函数
        self.Z1_Layer_start = Z1_Layer_start()
        self.IPM1 = IPM1()
        self.IPM2 = IPM2()

    def forward(self, rho1,mu1,v1,t,H1,H2,c):

        AA,b,A0 = self.Z1_Layer_start(rho1,mu1,v1,t,H2,c)
        x, y, z, A, b, c, normc = self.IPM1(AA,b,A0)
        for stage in range(100):
            x, y, z, A, b, c, normc, y1 = self.IPM2(x, y, z, A, b, c, normc)
            print('stage: {}'.format(stage + 1))

        x, y, z, A, b, c, normc, y1 = self.IPM2(x, y, z, A, b, c, normc)

        p1 = y1[0, 0] + y1[8, 0] + y1[15, 0] + y1[21, 0] + y1[26, 0] + y1[30, 0] + y1[33, 0] + y1[35, 0] + y1[
            36 + 0, 0] + y1[36 + 8, 0] + y1[36 + 15, 0] + y1[36 + 21, 0] + y1[36 + 26, 0] + y1[36 + 30, 0] + y1[
                 36 + 33, 0] + y1[36 + 35, 0]
        t121=y1[78,0]
        t122=y1[79,0]
        t211=y1[76,0]
        t212=y1[77,0]



        return p1,t121, t122, t211, t212

class Z2SDPSlover(torch.nn.Module):
    def __init__(self):
        super(Z2SDPSlover, self).__init__()  # 第一句话，调用父类的构造函数
        self.Z1_Layer_start = Z1_Layer_start()
        self.IPM1 = IPM1()
        self.IPM2 = IPM2()

    def forward(self, rho1,mu1,v1,t,H1,H2,c):

        AA,b,A0 = self.Z2_Layer_start(rho1,mu1,v1,t,H1,c)
        x, y, z, A, b, c, normc = self.IPM1(AA,b,A0)
        for stage in range(100):
            x, y, z, A, b, c, normc, y1 = self.IPM2(x, y, z, A, b, c, normc)
            print('stage: {}'.format(stage + 1))

        x, y, z, A, b, c, normc, y1 = self.IPM2(x, y, z, A, b, c, normc)

        p2 = y1[0,0]+y1[8,0]+y1[15,0]+y1[21,0]+y1[26,0]+y1[30,0]+y1[33,0]+y1[35,0]+y1[36+0,0]+y1[36+8,0]+y1[36+15,0]+y1[36+21,0]+y1[36+26,0]+y1[36+30,0]+y1[36+33,0]+y1[36+35,0]

        t121=y1[76,0]
        t122=y1[77,0]
        t211=y1[78,0]
        t212=y1[79,0]



        return p2,t121, t122, t211, t212

class Z2SDPSloveri(torch.nn.Module):
    def __init__(self):
        super(Z2SDPSloveri, self).__init__()  # 第一句话，调用父类的构造函数
        self.Z1_Layer_start = Z1_Layer_start()
        self.IPM1 = IPM1()
        self.IPM2 = IPM2()

    def forward(self, rho1, mu1, v1, t, H1, H2, c):
        AA, b, A0 = self.Z2_Layer_start(rho1, mu1, v1, t, H2, c)
        x, y, z, A, b, c, normc = self.IPM1(AA, b, A0)
        for stage in range(100):
            x, y, z, A, b, c, normc, y1 = self.IPM2(x, y, z, A, b, c, normc)
            print('stage: {}'.format(stage + 1))

        x, y, z, A, b, c, normc, y1 = self.IPM2(x, y, z, A, b, c, normc)

        p2 = y1[0, 0] + y1[8, 0] + y1[15, 0] + y1[21, 0] + y1[26, 0] + y1[30, 0] + y1[33, 0] + y1[35, 0] + y1[
            36 + 0, 0] + y1[36 + 8, 0] + y1[36 + 15, 0] + y1[36 + 21, 0] + y1[36 + 26, 0] + y1[36 + 30, 0] + y1[
                 36 + 33, 0] + y1[36 + 35, 0]

        t121 = y1[76, 0]
        t122 = y1[77, 0]
        t211 = y1[78, 0]
        t212 = y1[79, 0]

        return p2, t121, t122, t211, t212




        # model = Z1SDPSlover()
#
#
#
# rho1=torch.zeros(1,1).double()
# v1=torch.zeros(4,1).double()
# mu1=torch.zeros(1,1).double()
# t=torch.zeros(4,1).double()
# c=torch.tensor(1.).double()
# H_temp1 = sio.loadmat('D:\Python36/data_r.mat')
# H_temp2 = sio.loadmat('D:\Python36/data_i.mat')
# H1 = H_temp1["data_r"]
# H2 = H_temp2["data_i"]
# H1=torch.from_numpy(H1)
# H10=H1[0]
#
# y1=model(rho1,mu1,v1,t,H10,H10,c)


clc;
clear all;
warning('off')
%%%%%%%%%%%%%%%%%%%%%%%%
% Fig 4(a) Nc=2 ,K=2 ,Nt=8 ,ADMM
%%%%%%%%%%%%%%%%%%%%%%%%

Nt=8;%天线数
Nc=2;%小区数
k=2;%每个基站对应移动台数
N=Nc.^2.*k;%信道h的数量
gamma=10;%设定信噪比阈值γ
epsilon=0.05;%设定球形误差模型的误差半径ε
I=eye(Nt); %生成维度为Nt单位矩阵
Q=epsilon.^-2*I;
sigma=0.02;%噪声功率
% t=zeros(Nc*(Nc-1)*k,1);%所有的小区间干扰
% %???
rho1=0.002*1;
rho2=0.002*1;%松弛变量
mu1=0;
mu2=0;%p1=rho1的对偶变量
v1=0.002*ones(Nc*k,1);
v2=0.002*ones(Nc*k,1);%t1=E1*t的对偶变量
% % 
% rho1=0;
% rho2=0;%松弛变量
% mu1=0;
% mu2=0;%p1=rho1的对偶变量
% v1=zeros(Nc*k,1);
% v2=zeros(Nc*k,1);%t1=E1*t的对偶变量

% rho1=1;
% rho2=1;%松弛变量
% mu1=0;
% mu2=0;%p1=rho1的对偶变量
% v1=ones(Nc*k,1);
% v2=ones(Nc*k,1);%t1=E1*t的对偶变量
t=0.002*ones(Nc*(Nc-1)*k,1);%所有的小区间干扰

E1=[0 0 1 0;
    0 0 0 1;
    1 0 0 0;
    0 1 0 0];%线性映射矩阵
E2=[1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1];

H=rand(Nt,N);
 %c=10.^-6;%惩罚参数

q=30;%迭代次数
channelRealizationNum=1;%信道实现次数

phi11=zeros(Nt+1,Nt+1);
phi12=zeros(Nt+1,Nt+1);
phi21=zeros(Nt+1,Nt+1);
phi22=zeros(Nt+1,Nt+1);
psi121=zeros(Nt+1,Nt+1);
psi122=zeros(Nt+1,Nt+1);
psi211=zeros(Nt+1,Nt+1);
psi212=zeros(Nt+1,Nt+1);

T1={};
T2={};
P1={};
P2={};
T={};
RHO1={};
RHO2={};
V1={};
V2={};
MU1={};
MU2={};%用于记录每一次循环的结果
E=[E1.' E2.'].';

sumPower=0;
x_channelRealizationNum=[];
y_averageSumPower=[];

load('D:\Python36\P_out.mat','P_out')
load('D:\MCBF\data_X.mat','data_X')
for i=1:200
    for j=1:8
        for k1=1:8
            H(k1,j,i)=data_X((2*k1-1)+16*(j-1),i) + 1i*data_X((2*k1)+16*(j-1),i);

        end
        
    end
end



aa=50;


Z1_in_save=[];
Z1_out_save=[];
Z2_in_save=[];
Z2_out_save=[];
P_final_sum = [];

for c_iter =1%4567比较好
for l=1:channelRealizationNum
%     c=10^(l-7);
%     lamda = 10^((l-7));
    
%for i=1:l
    
%生成N个信道
% for j=1:N 
%    j;
%    h_head=sqrt(0.5)*randn(Nt,1)+1i*sqrt(0.5)*randn(Nt,1);   %产生服从复高斯随机的预定的信道向量
%    H(:,j)=h_head;%信道衰落*1.739*10^-4,将生成的信道放到信道矩阵中
% end  
%对生成信道进行编号，h111指第一个基站对第一个基站的第一个接收台的信道
% H=[-0.395397041799618 + 0.244938058674001i,...
%     0.356324713677101 - 0.788666312983587i,...
%     -0.125787286706314 + 0.560700579244636i,...
%     -0.910340522898403 + 0.745341438856100i,...
%     0.0974356839706305 - 0.591445695832183i,...
%     -0.580625359964030 - 0.431607108100838i,...
%     -0.169551269248824 - 0.174599895761527i,...
%     -0.992859826428695 + 0.373540317909920i;
%     -0.571939895117463 - 0.184649287996052i,-0.581581577007663 + 0.471810884508112i,1.25333443909358 - 1.49195318725676i,0.154209278094303 - 1.39985881328431i,-1.12205217240366 + 0.603140538870143i,-0.757951328546448 + 0.538093984151471i,0.651235059763151 + 0.312271899814662i,0.145355384026574 + 0.747340870191241i;0.820954003331967 - 0.130611747760376i,0.142115643629586 + 0.562385218772410i,-1.77399204747481 - 0.565257714446284i,-1.10773020099387 - 1.32044284842924i,-0.720644704557552 + 0.337524113313977i,-0.759497479698823 + 0.179908826375169i,-0.641261972403455 - 0.568239129931604i,-1.40798804537045 + 0.0104231899584576i;0.418681650989804 - 0.0719375080861488i,-0.712093671212722 + 0.733617525635027i,-0.322857309183956 + 0.400911932952920i,0.553863964875646 - 1.29567112052765i,-0.979508171807843 + 0.213772575083775i,0.614866686651010 - 0.108069729356097i,0.102134932176081 + 0.203370580198551i,0.0200881498198891 + 1.14180864183106i;0.171648519674210 - 0.627234222205841i,-0.433577768272249 - 0.0144470348370814i,1.71853643721188 - 0.00102028357028278i,-0.219606088489509 + 0.600045239077025i,0.675206670133200 + 0.293998163841826i,0.693708107903467 - 0.394149009767540i,0.987410293137466 + 0.330231661030196i,1.36508053671389 - 0.576896575993195i;0.170043260139469 + 0.524232772921012i,-0.465957482223376 + 0.437665924254470i,-0.333368381157086 + 0.441181362458285i,0.463384335300251 + 0.286510659240013i,-0.425056254136229 + 0.0303877932216818i,-1.24367734544113 + 0.478723776139856i,-0.134137279299292 + 0.00322554505435696i,0.665484205049906 - 1.41262902401687i;-0.580888502772182 + 0.984439626326138i,-0.589182107767211 + 1.27495843533764i,-0.396212075783006 + 0.0894071690434815i,-0.380046814689614 - 0.496711426937006i,-0.828651934421281 - 0.670940553324188i,-0.104719655861968 + 0.600543979366815i,-0.596682080611657 + 0.230580703847922i,-0.00489632647154977 - 0.529051218444872i;0.702236318923942 + 1.74931167059732i,0.267413145333260 + 0.0374719297725002i,-0.867243396892325 + 0.481446234361536i,0.232385979032570 + 1.05998560696109i,-0.408078611640754 + 0.382974948526602i,0.178149717184599 + 0.331850291268908i,1.37421822386300 + 0.851771435788070i,-0.239172962465960 - 0.140929224511207i];

% h111=H(:,1);
% h112=H(:,2);
% h221=H(:,3);
% h222=H(:,4);
% h121=H(:,5);
% h122=H(:,6);
% h211=H(:,7);
% h212=H(:,8);


% dmnk=0.035;
% psimu = 0;
% psisigma = 8;
% % psimnk=lognrnd(psimu,psisigma);
% psimnk=0.340234220818849;
% phimnkdBi=15;
% phimnk=10^(phimnkdBi/10);
% xishu = 10^(-(128.1+37.6*(log10(dmnk))) /20 );


Hmnk=H(:,:,1+aa);





% Hmnk=(Hmnk +Q) * xishu * psimnk * phimnk;

% h111=H(:,1,l+aa);
% h112=H(:,3,l+aa);
% h221=H(:,5,l+aa);
% h222=H(:,7,l+aa);
% h121=H(:,6,l+aa);
% h122=H(:,8,l+aa);
% h211=H(:,2,l+aa);
% h212=H(:,4,l+aa);


h111=Hmnk(:,1);
h112=Hmnk(:,3);
h221=Hmnk(:,5);
h222=Hmnk(:,7);
h121=Hmnk(:,6);
h122=Hmnk(:,8);
h211=Hmnk(:,2);
h212=Hmnk(:,4);



P_sum=[];
iter=[];

%%%%%%%%%%%%%%%%%%

%迭代q次

% c=5*(10^(3));
% c= 10^(c_iter-1);
% % c = 4800 + c_iter*100;
c=c_iter*1000+3000;
%c = 3000;
c2=c;
lamda = c;
for j=1:q 
    

        %基站1 Z层
        cvx_begin sdp quiet  %sdp:调用半确定编程模式;quiet:防止模型在解算时产生任何屏幕输出。cvx_begin sdp quiet会调用SDP模式并使求解器输出无效。
        variable W11(Nt,Nt) nonnegative semidefinite;
        variable W12(Nt,Nt) nonnegative semidefinite;
        variable lambda111; 
        variable lambda112;
        variable lambda121;
        variable lambda122;%表示λ，松弛变量
        variable t121;
        variable t122;
        variable t211;
        variable t212;%表示最差情况下小区间干扰
        variable t1(Nc*k,1)%[t211,t212,t121,t122]
        variable p1
        
        %式29a
        %pn=trace(W11)+trace(W12)
        %rho1为ρn，所以Σ(ρn-pn)^2=(rho1-(trace(W11)+trace(W12)))*(rho1-(trace(W11)+trace(W12)))
        
%         minimize (trace(W11)+trace(W12)+c/2.*(rho1-(trace(W11)+trace(W12)))*(rho1-(trace(W11)+trace(W12)))...
%         +c/2.*square_pos((norm(E1*t-[t211,t212,t121,t122]',2)))...
%     -v1.'*(E1*t-[t211,t212,t121,t122]')...
% -mu1.*(rho1-(trace(W11)+trace(W12)))); 
         minimize (trace(W11)+trace(W12) + (c/2)*((rho1-(trace(W11)+trace(W12)))*(rho1-(trace(W11)+trace(W12))))...
        +c/2.*square_pos((norm(E1*t-[t211,t212,t121,t122]',2)))...
    -v1.'*([t211,t212,t121,t122]')...
-mu1.*((trace(W11)+trace(W12)))); 
        subject to 
        phi11=[(1/gamma).*W11-W12+lambda111.*Q,((1/gamma).*W11-W12)* h111;h111'*((1/gamma).*W11-W12),h111'*((1/gamma).*W11-W12)*h111-lambda111-t211-sigma.^2]>=0;
        phi12=[(1/gamma).*W12-W11+lambda112.*Q,((1/gamma).*W12-W11)* h112;h112'*((1/gamma).*W12-W11),h112'*((1/gamma).*W12-W11)*h112-lambda112-t212-sigma.^2]>=0;
        psi121=[-(W11+W12)+lambda121*Q,-(W11+W12)*h121;-h121'*(W11+W12),-h121'*(W11+W12)*h121+t121-lambda121]>=0;
        psi122=[-(W11+W12)+lambda122*Q,-(W11+W12)*h122;-h122'*(W11+W12),-h122'*(W11+W12)*h122+t122-lambda122]>=0;
        lambda111>=0;
        lambda112>=0;
        lambda121>=0;
        lambda122>=0;
        W11>=0;
        W12>=0;
        t121>=0;
        t122>=0;
        t211>=0;
        t212>=0;
%         p1>0
        cvx_end
        t1211=t121;
        t1221=t122;
        t2111=t211;
        t2121=t212;
         
        t121=0;
        t122=0;
        t211=0;
        t212=0;
        
%          %基站1 Z层
%         cvx_begin sdp quiet  %sdp:调用半确定编程模式;quiet:防止模型在解算时产生任何屏幕输出。cvx_begin sdp quiet会调用SDP模式并使求解器输出无效。
%         variable W11(Nt,Nt) nonnegative semidefinite;
%         variable W12(Nt,Nt) nonnegative semidefinite;
%         variable lambda111; 
%         variable lambda112;
%         variable lambda121;
%         variable lambda122;%表示λ，松弛变量
%         variable t121;
%         variable t122;
%         variable t211;
%         variable t212;%表示最差情况下小区间干扰
%         variable t1(Nc*k,1)%[t211,t212,t121,t122]
%         variable t0
%         
%         
%         %式29a
%         %pn=trace(W11)+trace(W12)
%         %rho1为ρn，所以Σ(ρn-pn)^2=(rho1-(trace(W11)+trace(W12)))*(rho1-(trace(W11)+trace(W12)))
%         
% %         minimize (trace(W11)+trace(W12)+c/2.*(rho1-(trace(W11)+trace(W12)))*(rho1-(trace(W11)+trace(W12)))...
% %         +c/2.*square_pos((norm(E1*t-[t211,t212,t121,t122]',2)))...
% %     -v1.'*(E1*t-[t211,t212,t121,t122]')...
% % -mu1.*(rho1-(trace(W11)+trace(W12)))); 
%          minimize (trace(W11)+trace(W12) + (c/2)*((rho1-(trace(W11)+trace(W12)))*(rho1-(trace(W11)+trace(W12))))...
%        +c/2*t0...                            %+c/2.*square_pos((norm(E1*t-[t211,t212,t121,t122]',2)))...
%     -v1.'*([t211,t212,t121,t122]')...
% -mu1.*((trace(W11)+trace(W12)))); 
%         subject to 
%         phi11=[(1/gamma).*W11-W12+lambda111.*Q,((1/gamma).*W11-W12)* h111;h111'*((1/gamma).*W11-W12),h111'*((1/gamma).*W11-W12)*h111-lambda111-t211-sigma.^2]>=0;
%         phi12=[(1/gamma).*W12-W11+lambda112.*Q,((1/gamma).*W12-W11)* h112;h112'*((1/gamma).*W12-W11),h112'*((1/gamma).*W12-W11)*h112-lambda112-t212-sigma.^2]>=0;
%         psi121=[-(W11+W12)+lambda121*Q,-(W11+W12)*h121;-h121'*(W11+W12),-h121'*(W11+W12)*h121+t121-lambda121]>=0;
%         psi122=[-(W11+W12)+lambda122*Q,-(W11+W12)*h122;-h122'*(W11+W12),-h122'*(W11+W12)*h122+t122-lambda122]>=0;
%         lambda111>=0;
%         lambda112>=0;
%         lambda121>=0;
%         lambda122>=0;
%         W11>=0;
%         W12>=0;
%         t121>=0;
%         t122>=0;
%         t211>=0;
%         t212>=0;
%         t00=[t0,([t211,t212,t121,t122]'-E1*t)';
%             ([t211,t212,t121,t122]'-E1*t),ones(4)]>=0;
%         cvx_end
%         t1211=t121;
%         t1221=t122;
%         t2111=t211;
%         t2121=t212;
        
        %基站2 Z层
        cvx_begin sdp quiet  %sdp:调用半确定编程模式;quiet:防止模型在解算时产生任何屏幕输出。cvx_begin sdp quiet会调用SDP模式并使求解器输出无效。
        variable W21(Nt,Nt) nonnegative semidefinite;
        variable W22(Nt,Nt) nonnegative semidefinite;
        variable lambda211;
        variable lambda212;
        variable lambda221;
        variable lambda222;%表示λ，松弛变量
        variable t121;
        variable t122;
        variable t211;
        variable t212;%表示最差情况下小区间干扰
        variable t2(Nc*k,1)%[t121,t122,t211,t212]'
        variable p2
        %minimize (trace(W21)+trace(W22)+c/2.*(rho2-(trace(W21)+trace(W22)))*(rho2-(trace(W21)+trace(W22)))+c/2.*square_pos((norm(E2*t-[t121,t122,t211,t212]',2)))-v2.'*(E2*t-[t121,t122,t211,t212]')-mu2.*(rho2-(trace(W21)+trace(W22))));
        minimize (trace(W21)+trace(W22)...
        +c/2.*((rho2-(trace(W21)+trace(W22)))*(rho2-(trace(W21)+trace(W22))))...
        +c/2.*square_pos((norm(E2*t-[t121,t122,t211,t212]',2)))...
        -v2.'*([t121,t122,t211,t212]')...
        -mu2.*((trace(W21)+trace(W22))));
        
        subject to 
        phi21=[(1/gamma)*W21-W22+lambda221*Q,((1/gamma)*W21-W22)* h221;h221'*((1/gamma)*W21-W22),h221'*((1/gamma)*W21-W22)*h221-lambda221-t121-sigma.^2]>=0;
        phi22=[(1/gamma)*W22-W21+lambda222*Q,((1/gamma)*W22-W21)* h222;h222'*((1/gamma)*W22-W21),h222'*((1/gamma)*W22-W21)*h222-lambda222-t122-sigma.^2]>=0;
        psi211=[-(W21+W22)+lambda211*Q,-(W21+W22)*h211;-h211'*(W21+W22),-h211'*(W21+W22)*h211+t211-lambda211]>=0;
        psi212=[-(W21+W22)+lambda212*Q,-(W21+W22)*h212;-h212'*(W21+W22),-h212'*(W21+W22)*h212+t212-lambda212]>=0;
        lambda211>=0;
        lambda212>=0;
        lambda221>=0;
        lambda222>=0;
        t121>=0;
        t122>=0;
        t211>=0;
        t212>=0;
       % p2>0;
        cvx_end
        
        
%         %基站2 Z层
%         cvx_begin sdp quiet  %sdp:调用半确定编程模式;quiet:防止模型在解算时产生任何屏幕输出。cvx_begin sdp quiet会调用SDP模式并使求解器输出无效。
%         variable W21(Nt,Nt) nonnegative semidefinite;
%         variable W22(Nt,Nt) nonnegative semidefinite;
%         variable lambda211;
%         variable lambda212;
%         variable lambda221;
%         variable lambda222;%表示λ，松弛变量
%         variable t121;
%         variable t122;
%         variable t211;
%         variable t212;%表示最差情况下小区间干扰
%         variable t2(Nc*k,1)%[t121,t122,t211,t212]'
%         variable p2
%         %minimize (trace(W21)+trace(W22)+c/2.*(rho2-(trace(W21)+trace(W22)))*(rho2-(trace(W21)+trace(W22)))+c/2.*square_pos((norm(E2*t-[t121,t122,t211,t212]',2)))-v2.'*(E2*t-[t121,t122,t211,t212]')-mu2.*(rho2-(trace(W21)+trace(W22))));
%         minimize (trace(W21)+trace(W22)...
%         +c/2.*((rho2-(trace(W21)+trace(W22)))*(rho2-(trace(W21)+trace(W22))))...
%         +c/2*t0...                            %+c/2.*square_pos((norm(E1*t-[t211,t212,t121,t122]',2)))...
%         -v2.'*([t121,t122,t211,t212]')...
%         -mu2.*((trace(W21)+trace(W22))));
%         
%         subject to 
%         phi21=[(1/gamma)*W21-W22+lambda221*Q,((1/gamma)*W21-W22)* h221;h221'*((1/gamma)*W21-W22),h221'*((1/gamma)*W21-W22)*h221-lambda221-t121-sigma.^2]>=0;
%         phi22=[(1/gamma)*W22-W21+lambda222*Q,((1/gamma)*W22-W21)* h222;h222'*((1/gamma)*W22-W21),h222'*((1/gamma)*W22-W21)*h222-lambda222-t122-sigma.^2]>=0;
%         psi211=[-(W21+W22)+lambda211*Q,-(W21+W22)*h211;-h211'*(W21+W22),-h211'*(W21+W22)*h211+t211-lambda211]>=0;
%         psi212=[-(W21+W22)+lambda212*Q,-(W21+W22)*h212;-h212'*(W21+W22),-h212'*(W21+W22)*h212+t212-lambda212]>=0;
%         lambda211>=0;
%         lambda212>=0;
%         lambda221>=0;
%         lambda222>=0;
%         t121>=0;
%         t122>=0;
%         t211>=0;
%         t212>=0;
%         t00=[t0,([t121,t122,t211,t212]'-E2*t)';
%             ([t121,t122,t211,t212]'-E2*t),ones(4)]>=0;
%        % p2>0;
%         cvx_end
        
        p1=trace(W11)+trace(W12);
        p2=trace(W21)+trace(W22);
        t1=[t2111,t2121,t1211,t1221]';
        t2=[t121,t122,t211,t212]';
        T1{j}=t1;
        T2{j}=t2;
        P1{j}=p1;
        P2{j}=p2;
        P_sum=[P_sum,p1+p2];
        iter=[iter,j];
        
        
        %保存Z层数据
%         Z1_in = [real(h111'),imag(h111'),real(h112'),imag(h112'),...
%             real(h121'),imag(h121'),real(h122'),imag(h122'),t',rho1,v1',c];
        Z1_in = [t',rho1,v1',c];
%         Z1_out = [p1,t2111,t2121,t1211,t1221];
        Z1_out = [p1,t2111,t2121,t1211,t1221];
        Z1_in_save = [Z1_in_save,Z1_in'];
        Z1_out_save = [Z1_out_save,Z1_out'];
        
%         Z2_in = [real(h221'),imag(h221'),real(h222'),imag(h222'),...
%             real(h211'),imag(h211'),real(h212'),imag(h212'),t',rho2,v2',c];
%         Z2_out = [p2,t121,t122,t211,t212];
        Z2_in = [t',rho2,v2',c];
        Z2_out = [p2,t121,t122,t211,t212,];
        Z2_in_save = [Z2_in_save,Z2_in'];
        Z2_out_save = [Z2_out_save,Z2_out'];
        
%         % X层 原文
        t=(pinv(E))*([t1.' t2.'].'-(1/c2)*[v1.' v2.'].');
        rho1=p1-(1/c2)*mu1;
        rho2=p2-(1/c2)*mu2;
        T{j}=t;
        RHO1{j}=rho1;
        RHO2{j}=rho2;

        % X层 CVX
          %基站1
%           v1h=v1.';
%           v2h=v2.';
%         cvx_begin sdp quiet
%         variable t(Nc*(Nc-1)*k,1);
%         minimize c/2.*square_pos((norm(E1*t-t1,1)))+c/2.*square_pos((norm(E2*t-t2,1)))+v1h*E1*t+v2h*E2*t
%         cvx_end
%         
%         cvx_begin sdp quiet
%         variable rho1;
%         variable rho2;
%         minimize c/2.*((rho1-p1).^2+(rho2-p2).^2)+mu1*rho1+mu2*rho2;
%         cvx_end
        % M层
        v1=v1+lamda*(E1*t-t1);
        mu1=mu1+lamda*(rho1-p1);
        v2=v2+lamda*(E2*t-t2);
        mu2=mu2+lamda*(rho2-p2);
        V1{j}=v1;
        MU1{j}=mu1;
        %更新步长c
%         if j>10&&c>300
%             c=c-200;
%         end
%         c2=c;
%         lamda = c;

end
    sumPower=sumPower+10*log10(1000*(p1+p2));
    %初始化
% t=zeros(Nc*(Nc-1)*k,1);%所有的小区间干扰
% rho1=0;
% rho2=0;%松弛变量
% mu1=0;
% mu2=0;%p1=rho1的对偶变量
% v1=zeros(Nc*k,1);
% v2=zeros(Nc*k,1);%t1=E1*t的对偶变量

rho1=1;
rho2=1;%松弛变量
mu1=0;
mu2=0;%p1=rho1的对偶变量
v1=ones(Nc*k,1);
v2=ones(Nc*k,1);%t1=E1*t的对偶变量
t=ones(Nc*(Nc-1)*k,1);%所有的小区间干扰
    
%end
    x_channelRealizationNum(l)=l;
    y_averageSumPower(l)=sumPower;
    sumPower=0;
    PP1(:,l)=P1;
    PP2(:,l)=P2;
    
   
    P_final=abs((P_sum-P_out(3,l+aa)))/P_out(3,l+aa);
    P_final_sum(l+aa,:,c_iter)= P_final;
    
end

for iternum=1:q
    Last(c_iter,iternum) = sum(P_final_sum(:,iternum))/channelRealizationNum;
end
% figure(4)        
% plot(x_channelRealizationNum,y_averageSumPower,'-*');
% legend('10 iterations','Pusa Early Drawf','Ife No.1','Location','northwest');
%xlabel('Channel realization number');ylabel('Average Sum Power (dBm)');



% praw = 0.0090073286430777;
% 
% P_final=abs((P_sum-praw))/praw;
% 
% P_final_sum(c_iter,:)= P_final;

end

% figure
% semilogy(iter,abs((P_sum-praw))/praw,'-')
% xlabel('迭代次数（q）');ylabel('平均精度');

% 数据储存
% save('Z1_in_save.mat','Z1_in_save');   
% save('Z1_out_save.mat','Z1_out_save');  
% save('Z2_in_save.mat','Z2_in_save');  
% save('Z2_out_save.mat','Z2_out_save');  
% 
% 
% 
% Z_in_save=[Z1_in_save,Z2_in_save];
% Z_in_save = Z_in_save(1:73,:);
% Z_in_save = Z_in_save';
% Z_out_save=[Z1_out_save,Z2_out_save]';
% save('Z_in_save.mat','Z_in_save');   
% save('Z_out_save.mat','Z_out_save');  



%%%%%%%%%%%%%%%%%%


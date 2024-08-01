
function H = H(N,Nt)

for j=1:N 
   j;
   h_head=sqrt(0.5)*randn(Nt,1)+1i*sqrt(0.5)*randn(Nt,1);   %产生服从复高斯随机的预定的信道向量
   H(:,j)=h_head;%信道衰落*1.739*10^-4,将生成的信道放到信道矩阵中
end  

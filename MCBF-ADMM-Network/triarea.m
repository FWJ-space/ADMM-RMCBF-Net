E1=[0 0 1 0;
    0 0 0 1;
    1 0 0 0;
    0 1 0 0];%����ӳ�����
E2=[1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1];
E=[E1.' E2.'].';
EE = (inv(E'*E)*E');
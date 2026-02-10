% clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
S_0=1;  % initial asset price
S_max=S_0;
Y_0=log(S_0);  % initial log-price
 
T=1;  % maturity
 
% Regime Switching BS model
sigma=[0.2 0.4];

Lambda=[-0.75, 0.75; 0.25, -0.25];
 
alpha=0.25;
a=-log(1-alpha);  % drawdown level
 
n_drawdown=4;
 
%% CTMC approximation
n_a=640;
h=a/n_a;
upper_bound=0.6;
n_half=ceil(upper_bound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=length(y_CTMC);
 
 
 
n_v=2;
 
% Construction of transition rate matrix
G_Y=zeros(n_grid,n_grid,n_v);

for k_v=1:n_v
    G_Y_0=[0,-sigma(k_v)^2/h^2*ones(1,n_grid-2),0];
    G_Y_1=[0,(r-d-sigma(k_v)^2/2)/(2*h)*ones(1,n_grid-2)+sigma(k_v)^2/(2*h^2)*ones(1,n_grid-2)];
    G_Y_2=[-(r-d-sigma(k_v)^2/2)/(2*h)*ones(1,n_grid-2)+sigma(k_v)^2/(2*h^2)*ones(1,n_grid-2),0];
    G_Y(:,:,k_v)=diag(G_Y_0)+diag(G_Y_1,1)+diag(G_Y_2,-1);
end
 
%% Laplace transform
A=15;k_1=10;k_2=10;
q=A/(2*T)+(0:k_1+k_2)*pi*1i/T;  % Laplace inversion grid
n_q=length(q);
 
B_m=[-G_Y(n_half-n_a+2:n_half+1,n_half-n_a+1,1),zeros(n_a,1);zeros(n_a,1),-G_Y(n_half-n_a+2:n_half+1,n_half-n_a+1,2)];
B_p=[-G_Y(n_half-n_a+2:n_half+1,n_half+2,1),zeros(n_a,1);zeros(n_a,1),-G_Y(n_half-n_a+2:n_half+1,n_half+2,2)];

B_p_inf=[-G_Y(1:n_half+n_a,n_half+n_a+1,1),zeros(n_half+n_a,1);zeros(n_half+n_a,1),-G_Y(1:n_half+n_a,n_half+n_a+1,2)];



Lambda_G=kron(Lambda,eye(n_a));

Lambda_G_inf=kron(Lambda,eye(n_half+n_a));

J_q=zeros(1,n_q);

for i=1:n_q
    A_G=zeros(n_a*n_v);
    A_G(1:n_a,1:n_a)=G_Y(n_half-n_a+2:n_half+1,n_half-n_a+2:n_half+1,1)-(q(i)+r)*eye(n_a);
    A_G(n_a+1:end,n_a+1:end)=G_Y(n_half-n_a+2:n_half+1,n_half-n_a+2:n_half+1,2)-(q(i)+r)*eye(n_a);
    A_Lambda_G=sparse(A_G+Lambda_G);

    A_G_inf=zeros(n_v*(n_half+n_a));
    A_G_inf(1:n_half+n_a,1:n_half+n_a)=G_Y(1:n_half+n_a,1:n_half+n_a,1)-(q(i)+r)*eye(n_half+n_a);
    A_G_inf(n_half+n_a+1:end,n_half+n_a+1:end)=G_Y(1:n_half+n_a,1:n_half+n_a,2)-(q(i)+r)*eye(n_half+n_a);
    A_Lambda_G_inf=sparse(A_G_inf+Lambda_G_inf);


    P_left=A_Lambda_G\B_m;
    P_right=A_Lambda_G\B_p;

    P_right_inf=A_Lambda_G_inf\B_p_inf;

    S_G_left=[P_left(n_a,:);P_left(end,:)];
    S_G_right=[P_right(n_a,:);P_right(end,:)];

    S_G_right_inf=[P_right_inf(n_half+1,:);P_right_inf(2*n_half+n_a+1,:)];

    B_G=sum(S_G_left,2);

    S_G=(eye(n_v)-S_G_left*S_G_right_inf-S_G_right)\B_G;

    J_q(i)=S_G(1);
end



% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;

Prob=real(J_q./q)*coe_inv



% V1=0.489799682097876;V2=0.499763831419919;V3=0.504912846445203;V4=0.507529344384109;V5=0.508848120772000;V6=0.509510143895065;
% benchmark=0.510173924676320;
% 1/21*(64*V4-56*V3+14*V2-V1)
% 1/315*(1024*V6-960*V5+280*V4-30*V3+V2)

toc;

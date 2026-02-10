% clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
S_0=1;  % initial asset price
Y_0=log(S_0);  % initial log-price
 
T=0.1;  % maturity
 
% Regime Switching BS model
sigma=[0.2 0.4];

Lambda=[-0.75, 0.75; 0.25, -0.25];
 
a=0.2;  % drawdown level

xi=0.1;  % drawup level

%% CTMC approximation
n_a=640;
h=a/n_a;
upper_bound=0.2;
n_half=ceil(upper_bound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=length(y_CTMC);

n_xi=floor(xi/h)+1;
 
n_v=2;
 
% Construction of transition rate matrix
G_Y=zeros(n_grid,n_grid,n_v);

for k_v=1:n_v
    G_Y_0=[0,-sigma(k_v)^2/h^2*ones(1,n_grid-2),0];
    G_Y_1=[0,(r-d-sigma(k_v)^2/2)/(2*h)*ones(1,n_grid-2)+sigma(k_v)^2/(2*h^2)*ones(1,n_grid-2)];
    G_Y_2=[-(r-d-sigma(k_v)^2/2)/(2*h)*ones(1,n_grid-2)+sigma(k_v)^2/(2*h^2)*ones(1,n_grid-2),0];
    G_Y(:,:,k_v)=diag(G_Y_0)+diag(G_Y_1,1)+diag(G_Y_2,-1);
end

% G=kron(G_v,eye(n_grid))+G_Y;
% G_Y_1=G_Y(:,:,1);
% G_Y_2=G_Y(:,:,2);
 
%% Laplace transform
A=15;k_1=10;k_2=10;
q=A/(2*T)+(0:k_1+k_2)*pi*1i/T;  % Laplace inversion grid
n_q=length(q);
 
% Recursion
C_q=zeros(n_grid,n_v,n_q);

% C_q(end,:)=1;

Indicate=diag([ones(n_a-n_xi,1);zeros(n_xi,1)]);


B_m=[-G_Y(n_half-n_a+2:n_half+1,n_half+1-n_a,1);-G_Y(n_half-n_a+2:n_half+1,n_half+1-n_a,2)];
B_p=[-G_Y(n_half-n_a+2:n_half+1,n_half+2,1),zeros(n_a,1);zeros(n_a,1),-G_Y(n_half-n_a+2:n_half+1,n_half+2,2)];

Lambda_G=kron(Lambda,eye(n_a));
for i=1:n_q
    A_G=zeros(n_a*n_v,n_a*n_v);
    A_G(1:n_a,1:n_a)=G_Y(n_half-n_a+2:n_half+1,n_half-n_a+2:n_half+1,1)-(q(i)*Indicate+r*eye(n_a));
    A_G(n_a+1:end,n_a+1:end)=G_Y(n_half-n_a+2:n_half+1,n_half-n_a+2:n_half+1,2)-(q(i)*Indicate+r*eye(n_a));
    A_Lambda_G=sparse(A_G+Lambda_G);
    P_left=A_Lambda_G\B_m;
    R_right=A_Lambda_G\B_p;

    S_left=[P_left(n_a);P_left(end)];
    S_right=[R_right(n_a,:);R_right(end,:)];
    C_q(n_half+1,:,i)=(eye(2)-S_right)\S_left;
end

h_a=reshape(C_q(n_half+1,1,:),1,n_q);


 
%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;
 
%% Probability/Digital option
 
Prob=real(h_a./q)*coe_inv

% V1=0.448940081764099;V2=0.419808177836303;V3=0.405861472598168;V4=0.399054791760366;V5=0.395694490654067;V6=0.394025263733110;
% benchmark=0.392363380087600;
% 1/21*(64*V5-56*V4+14*V3-V2)
% 1/315*(1024*V6-960*V5+280*V4-30*V3+V2)

toc;

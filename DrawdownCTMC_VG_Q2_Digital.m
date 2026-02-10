clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
S_0=1;  % initial asset price
Y_0=log(S_0);  % initial log-price

T=0.5;

% VG model
theta=-2.206;
sigma=0.962;
nu=0.00254;
C_para=1/nu;
G_para=sqrt(theta^2/sigma^4+2/(sigma^2*nu))+theta/sigma^2;
M_para=sqrt(theta^2/sigma^4+2/(sigma^2*nu))-theta/sigma^2;

a=0.5;  % drawdown level
 
xi=0.2;  % drawup level
%% CTMC approximation
n_a=160;
h=a/n_a;
upperbound=6.8;
n_half=ceil(upperbound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=2*n_half+1;


% % Construction of transition rate matrix
% sigma_bar2=C_para/M_para*((1-exp(-1/2*M_para*h))/M_para-1/2*h*exp(-1/2*M_para*h))+C_para/G_para*((1-exp(-1/2*G_para*h))/G_para-1/2*h*exp(-1/2*G_para*h));
% 
% % Gauss-Legendre quadrature for LAMBDA
% [nodes,weights]=lgwt(5,-1/2*h,1/2*h);
% 
% LAMBDA=zeros(2*n_grid-1,1);
% nodes1=(-n_grid+1:-1)*h+nodes;
% nodes2=(1:n_grid-1)*h+nodes;
% weights1=weights;
% 
% LAMBDA(1:n_grid-1)=-C_para./nodes1.'.*exp(G_para*nodes1.')*weights1;
% LAMBDA(n_grid+1:end)=C_para./nodes2.'.*exp(-M_para*nodes2.')*weights1;
% 
% mu=r-d+log(1-theta*nu-sigma^2*nu/2)/nu;
% 
% c_G=[0;-(mu)/(2*h)+(sigma_bar2)/(2*h^2)+LAMBDA(n_grid-1);LAMBDA(n_grid-2:-1:1)];  % first column of G, G is a Toeplitz matrix
% r_G=[0;(mu)/(2*h)+(sigma_bar2)/(2*h^2)+LAMBDA(n_grid+1);LAMBDA(n_grid+2:end)];  % first row of G
% G=toeplitz(c_G,r_G);
% G=G-diag(sum(G,2));
% G(1,:)=zeros(1,n_grid);
% G(end,:)=zeros(1,n_grid);
% % p=expm(G*T);  % transition probability

mu=r-d+log(1-theta*nu-sigma^2*nu/2)/nu;
sigma_bar2=C_para/M_para*((1-exp(-1/2*M_para*h))/M_para-1/2*h*exp(-1/2*M_para*h))+C_para/G_para*((1-exp(-1/2*G_para*h))/G_para-1/2*h*exp(-1/2*G_para*h));
G_D_u=mu/h+sigma_bar2/(2*h^2);
G_D_l=sigma_bar2/(2*h^2);
G_D_diag=-mu/h-sigma_bar2/h^2;
G_D=diag(G_D_u*ones(n_grid-1,1),1)+diag(G_D_diag*ones(n_grid,1))+diag(G_D_l*ones(n_grid-1,1),-1);

% Gauss-Legendre quadrature for LAMBDA
[nodes,weights]=lgwt(5,-1/2*h,1/2*h);
LAMBDA=zeros(2*n_grid-1,1);
nodes1=(-n_grid+1:-1)*h+nodes;
nodes2=(1:n_grid-1)*h+nodes;
weights1=weights;
LAMBDA(1:n_grid-1)=-C_para./nodes1.'.*exp(G_para*nodes1.')*weights1;
LAMBDA(n_grid+1:end)=C_para./nodes2.'.*exp(-M_para*nodes2.')*weights1;
c_G=LAMBDA(n_grid:-1:1);
r_G=LAMBDA(n_grid:end);
G_J=toeplitz(c_G,r_G);
G_J(end:-1:2,1)=cumsum(LAMBDA(1:n_grid-1));
G_J(end-1:-1:1,end)=cumsum(LAMBDA(n_grid+1:end),'reverse');
G_J=G_J-diag(sum(G_J,2));

G=G_D+G_J;
G(1,:)=0;
G(end,:)=0;
%% Laplace transform
A=15;k_1=10;k_2=10;
q=A/(2*T)+(0:k_1+k_2)*pi*1i/T;  % Laplace inversion grid
n_q=length(q);


% Recursion

B_q=zeros(n_grid,n_q);

% B_q(end,:)=1;
Indicate=y_CTMC<xi;

for k=n_grid-1:-1:n_half+1
    B_m=-G(k-n_a+1:k,1:k-n_a)*ones(k-n_a,1);
    B_p=-G(k-n_a+1:k,k+1:end)*B_q(k+1:end,:);
    for i=1:n_q
        A_G=G(k-n_a+1:k,k-n_a+1:k)-diag(q(i)*Indicate(k-n_a+1:k)+r);
        P_S=A_G\(B_m+B_p(:,i));
        B_q(k,i)=P_S(end);
    end
end

h_a=B_q(n_half+1,:);




%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;
 
%% Probability/Digital option
 
Prob=real(h_a./q)*coe_inv

% V1=0.984408243914253;V2=0.980559014808133;V3=0.978141740675362;V4=0.976925084452200;V5=0.976348660991276;V6=0.976074628175986;
% 1/21*(64*V5-56*V4+14*V3-V2)

toc;

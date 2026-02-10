clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
S_0=1;  % initial asset price
S_max=S_0;
Y_0=log(S_0);  % initial log-price

T=1;  % maturity
 
% VG model
theta=-2.206;
sigma=0.962;
nu=0.00254;
C_para=1/nu;
G_para=sqrt(theta^2/sigma^4+2/(sigma^2*nu))+theta/sigma^2;
M_para=sqrt(theta^2/sigma^4+2/(sigma^2*nu))-theta/sigma^2;
 
% a=0.8;  % drawdown level
alpha=0.5;
a=-log(1-alpha);

% n_drawdown=10;

%% CTMC approximation
n_a=640;
h=a/n_a;
% upperbound=0.6941;
% n_half=ceil(upperbound/h);
% y_CTMC=(-n_half:n_half)*h;
% n_grid=2*n_half+1;
y_CTMC=(-n_a:1)*h;
n_grid=length(y_CTMC);

% % Construction of transition rate matrix
% sigma_bar2=C_para/M_para*((1-exp(-1/2*M_para*h))/M_para-1/2*h*exp(-1/2*M_para*h))+C_para/G_para*((1-exp(-1/2*G_para*h))/G_para-1/2*h*exp(-1/2*G_para*h));
% 
% 
% 
% % Gauss-Legendre quadrature for LAMBDA
% [nodes,weights]=lgwt(5,-1/2*h,1/2*h);
% 
% LAMBDA=zeros(2*n_grid-1,1);
% nodes1=(-n_grid+1:-1)*h+nodes;
% nodes2=(1:n_grid-1)*h+nodes;
% weights1=weights;
% 
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
%  
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
 
%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;
 
B_m=-G(2:end-1,1);
B_p=-G(2:end-1,end);
Q_a=zeros(1,n_q);
for i=1:n_q
    A_G=G(2:end-1,2:end-1)-(q(i)+r)*eye(n_a);
    P_left=A_G\B_m;
    P_right=A_G\B_p;
    Q_a(i)=P_left(end)/(1-P_right(end));
end
h_a=Q_a./(1-Q_a);
Prob=real(h_a./q)*coe_inv

% % Recursion
% h_q=zeros(n_grid,n_q);
% h_q_last=ones(n_grid,n_q);
% Prob=0;
% for n=1:n_drawdown
%     for k=n_grid-1:-1:n_a+1
%         B_m=-G(k-n_a+1:k,1:k-n_a)*h_q_last(1:k-n_a,:);
%         B_p=-G(k-n_a+1:k,k+1:end)*h_q(k+1:end,:);
%         for i=1:n_q
%             A_G=G(k-n_a+1:k,k-n_a+1:k)-(q(i)+r)*eye(n_a);
%             P_S=A_G\(B_m(:,i)+B_p(:,i));
%             h_q(k,i)=P_S(end);
%         end
%     end
%     for k=n_a:-1:1
%         B_p=-G(1:k,k+1:end)*h_q_last(k+1:end,:);
%         for i=1:n_q
%             A_G=G(1:k,1:k)-q(i)*eye(k);
%             P_S=A_G\B_p(:,i);
%             h_q(k,i)=P_S(end);
%         end
%     end
%     h_q_last=h_q;
%     Prob=Prob+real(h_q(n_half+1,:)./q)*coe_inv;
% end
%  
% 
% Prob

% %% Parallel computing
% h_q=zeros(n_grid,n_q);
% h_q_last=ones(n_grid,n_q);
% h_a=zeros(n_drawdown,n_q);
% 
% parfor i=1:n_q
%     h_last_i=h_q_last(:,i);
%     h_i=h_q(:,i);
%     q_i=q(i);
%     G_temp=G;
%     A_G=G_temp(n_grid-n_a:n_grid-1,n_grid-n_a:n_grid-1)-(q_i+r)*eye(n_a);
%     A_G_inv=A_G\eye(n_a);
%     for n=1:n_drawdown
%         for k=n_grid-1:-1:n_a+1
%             B_m=-G_temp(k-n_a+1:k,1:k-n_a)*h_last_i(1:k-n_a);
%             B_p=-G_temp(k-n_a+1:k,k+1:end)*h_i(k+1:end);
% %             A_G=G_temp(k-n_a+1:k,k-n_a+1:k)-(q_i+r)*eye(n_a);
%             P_S=A_G_inv*(B_m+B_p);
%             h_i(k)=P_S(end);
%         end
%         for k=n_a:-1:1
%             B_p=-G_temp(1:k,k+1:end)*h_last_i(k+1:end);
%             A_G=G_temp(1:k,1:k)-(q_i+r)*eye(k);
%             P_S=A_G\B_p;
%             h_i(k)=P_S(end);
%         end
%         h_last_i=h_i;
%         h_a(n,i)=h_i(n_half+1);
%     end
% end
% 
% 
% Prob=sum(real(h_a./q)*coe_inv)



% V1=2.244066121222627;V2=2.074722449875579;V3=1.984640319844452;V4=1.943888985402627;V5=1.925836094789574;V6=1.917630621352926;V7=1.913768981584073;V8=1.911900770140597;V9=1.910982045677806;
% 1/21*(64*V6-56*V5+14*V4-V3)
% 1/315*(1024*V9-960*V8+280*V7-30*V6+V5)
 
 
toc;

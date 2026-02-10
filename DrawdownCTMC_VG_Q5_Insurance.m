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
 
% a=0.2;  % drawdown level
 
alpha=0.5;
a=-log(1-alpha);

% n_drawdown=4;
 
%% CTMC approximation
n_a=640;
h=a/n_a;
upperbound=2.5;
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

% Recursion (simplified)
J_q_diag=zeros(1,n_q);
for i=1:n_q
    B_p_inf=-sum(G(1:n_half,n_half+1:end),2);
    A_G_inf=G(1:n_half,1:n_half)-(q(i)+r)*eye(n_half);
    P_right_inf=A_G_inf\B_p_inf;
    
    B_m_den=-G(n_half-n_a+2:n_half+1,1:n_half-n_a+1)*P_right_inf(1:n_half-n_a+1);
    B_m_num=-sum(G(n_half-n_a+2:n_half+1,1:n_half-n_a+1),2);
    B_p=-sum(G(n_half-n_a+2:n_half+1,n_half+2:end),2);
    A_G=G(n_half-n_a+2:n_half+1,n_half-n_a+2:n_half+1)-(q(i)+r)*eye(n_a);
    P_num=A_G\B_m_num;
    P_den=A_G\(B_p+B_m_den);
    
    J_q_diag(i)=P_num(end)/(1-P_den(end));
end
Prob=real(J_q_diag./q)*coe_inv
 
% % Recursion
% J_q=zeros(n_grid,n_q,n_grid);
% J_q_last=ones(n_grid,n_q,n_grid);
% J_q_diag=zeros(n_grid,n_q);
% J_q_diag(end,:)=1;
% Prob=0;
% for n=1:n_drawdown
%     for k_y=n_grid-1:-1:n_a+1
%         B_p=-G(k_y-n_a+1:k_y,k_y+1:end)*J_q_diag(k_y+1:end,:);
%         B_m=-G(k_y-n_a+1:k_y,1:k_y-n_a)*J_q_last(1:k_y-n_a,:,k_y);
%         for i=1:n_q
%             A_G=G(k_y-n_a+1:k_y,k_y-n_a+1:k_y)-(q(i)+r)*eye(n_a);
%             P_S=A_G\(B_p(:,i)+B_m(:,i));
%             J_q_diag(k_y,i)=P_S(end);
%             
%             B_p_inf=-G(1:k_y-1,k_y:end)*J_q_diag(k_y:end,i);
%             A_G_inf=G(1:k_y-1,1:k_y-1)-(q(i)+r)*eye(k_y-1);
%             J_q(1:k_y-1,i,k_y)=A_G_inf\B_p_inf;
%         end
%         J_q(k_y,:,k_y)=J_q_diag(k_y,:);
%     end
%     J_q_last=J_q;
%     Prob=Prob+real(J_q(n_half+1,:,n_half+1)./q)*coe_inv
% end

% % Parallel computing
% J_q_last=ones(n_grid,n_q,n_grid);
% J_q_diag=zeros(n_grid,n_q);
% J_q_diag(end,:)=1;
% J_q=zeros(n_drawdown,n_q);
% parfor i=1:n_q
%     G_temp=G;
%     J_diag_i=J_q_diag(:,i);
%     J_last_i=J_q_last(:,i,:);
%     q_i=q(i);
%     A_G=G_temp(n_grid-n_a:n_grid-1,n_grid-n_a:n_grid-1)-(q_i+r)*eye(n_a);
%     A_G_inv=A_G\eye(n_a);
%     for n=1:n_drawdown
%         for k_y=n_grid-1:-1:n_a+1
%             B_p=-G_temp(k_y-n_a+1:k_y,k_y+1:end)*J_diag_i(k_y+1:end);
%             B_m=-G_temp(k_y-n_a+1:k_y,1:k_y-n_a)*J_last_i(1:k_y-n_a,k_y);
% %             A_G=G_temp(k_y-n_a+1:k_y,k_y-n_a+1:k_y)-(q_i+r)*eye(n_a);
%             P_S=A_G_inv*(B_p+B_m);
%             J_diag_i(k_y)=P_S(end);
%             
%             B_p_inf=-G_temp(1:k_y-1,k_y:end)*J_diag_i(k_y:end);
%             A_G_inf=G_temp(1:k_y-1,1:k_y-1)-(q_i+r)*eye(k_y-1);
%             J_last_i(1:k_y-1,k_y)=A_G_inf\B_p_inf;
%             J_last_i(k_y,k_y)=J_diag_i(k_y);
%         end
%         for k_y=n_a:-1:1
%             B_p=-G_temp(1:k_y,k_y+1:end)*J_diag_i(k_y+1:end);
%             A_G=G_temp(1:k_y,1:k_y)-(q_i+r)*eye(k_y);
%             P_S=A_G\B_p;
%             J_diag_i(k_y)=P_S(end);
%             
%             B_p_inf=-G_temp(1:k_y-1,k_y:end)*J_diag_i(k_y:end);
%             A_G_inf=G_temp(1:k_y-1,1:k_y-1)-(q_i+r)*eye(k_y-1);
%             J_last_i(1:k_y-1,k_y)=A_G_inf\B_p_inf;
%             J_last_i(k_y,k_y)=J_diag_i(k_y);
%         end
%         J_q(n,i)=J_last_i(n_half+1,n_half+1);
%     end
% end
% Prob=sum(real(J_q./q)*coe_inv)   



% V9=0.983432916133906;V8=0.983674124848757;V7=0.984165806249254;V6=0.985186050927836;V5=0.987367376806354;V4=0.992217668643069;V3=1.003393869791865;V2=1.029307858438707;V1=1.085399111656937;
% 1/21*(64*V6-56*V5+14*V4-V3)
% 1/315*(1024*V9-960*V8+280*V7-30*V6+V5)
 

 
 
toc;

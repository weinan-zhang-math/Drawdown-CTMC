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
 
% Kou model
sigma=0.3;  % volatility
p_p=0.5;
p_m=1-p_p;
eta_p=10;
eta_m=10;
lambda=3;
 
% a=0.05;  % drawdown level
alpha=0.25;
a=-log(1-alpha);

% n_drawdown=3;
 
%% CTMC approximation
n_a=160;
h=a/n_a;
upperbound=1.5;
n_half=ceil(upperbound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=2*n_half+1;


% % Construction of transition rate matrix
% sigma_bar2=lambda*(p_p*(2/eta_p^2-(1/4*h^2+1/eta_p*h+2/eta_p^2)*exp(-1/2*eta_p*h))+p_m*(2/eta_m^2-(1/4*h^2+1/eta_m*h+2/eta_m^2)*exp(-1/2*eta_m*h)));
% LAMBDA=zeros(2*n_grid-1,1);
% LAMBDA(1:n_grid-1)=lambda*p_m*(exp(((-n_grid+1:-1)+1/2)*h*eta_m)-exp(((-n_grid+1:-1)-1/2)*h*eta_m));
% LAMBDA(n_grid+1:end)=lambda*p_p*(exp(-((1:n_grid-1)-1/2)*h*eta_p)-exp(-((1:n_grid-1)+1/2)*h*eta_p));
% 
% k=floor(1/h-1/2);  % number of intervals within [1/2h,1]
% LAMBDA_k=[lambda*p_m*(exp(((-k:-1)+1/2)*h*eta_m)-exp(((-k:-1)-1/2)*h*eta_m)),0,lambda*p_p*(exp(-((1:k)-1/2)*h*eta_p)-exp(-((1:k)+1/2)*h*eta_p))];
% mu_bar=(-k:k)*h*LAMBDA_k.'+(k+1)*h*lambda*(p_p*(exp(-(k+1-1/2)*h*eta_p)-exp(-eta_p))-p_m*(exp(-(k+1-1/2)*h*eta_m)-exp(-eta_m)));
% 
% 
% mu=r-d-sigma^2/2-lambda*(p_p*eta_p/(eta_p-1)+p_m*eta_m/(eta_m+1)-1);
% 
% c_G=[0;-(mu-mu_bar)/(2*h)+(sigma^2+sigma_bar2)/(2*h^2)+LAMBDA(n_grid-1);LAMBDA(n_grid-2:-1:1)];  % first column of G, G is a Toeplitz matrix
% r_G=[0;(mu-mu_bar)/(2*h)+(sigma^2+sigma_bar2)/(2*h^2)+LAMBDA(n_grid+1);LAMBDA(n_grid+2:end)];  % first row of G
% G=toeplitz(c_G,r_G);
% G=G-diag(sum(G,2));
% G(1,:)=zeros(1,n_grid);
% G(end,:)=zeros(1,n_grid);
%  
% % p=expm(G*T);  % transition probability
 
mu=r-d-sigma^2/2-lambda*(p_p*eta_p/(eta_p-1)+p_m*eta_m/(eta_m+1)-1);
mu_bar=lambda*p_p*(-exp(-eta_p)+(1-exp(-eta_p))/eta_p)-lambda*p_m*(-exp(-eta_m)+(1-exp(-eta_m))/eta_m);
G_D_u=(mu-mu_bar)/(2*h)+sigma^2/(2*h^2);
G_D_l=-(mu-mu_bar)/(2*h)+sigma^2/(2*h^2);
G_D_diag=-sigma^2/h^2;
G_D=diag(G_D_diag*ones(n_grid,1))+diag(G_D_u*ones(n_grid-1,1),1)+diag(G_D_l*ones(n_grid-1,1),-1);
G_D(1,:)=0;
G_D(end,:)=0;

G_J=zeros(n_grid,n_grid);
LAMBDA=zeros(2*n_grid-1,1);
LAMBDA(1:n_grid-1)=lambda*p_m*(exp(((-n_grid+1:-1)+1/2)*h*eta_m)-exp(((-n_grid+1:-1)-1/2)*h*eta_m));
LAMBDA(n_grid+1:end)=lambda*p_p*(exp(-((1:n_grid-1)-1/2)*h*eta_p)-exp(-((1:n_grid-1)+1/2)*h*eta_p));
LAMBDA(n_grid)=-lambda*p_m*exp(-1/2*h*eta_m)-lambda*p_p*exp(-1/2*h*eta_p);
c_G=LAMBDA(n_grid:-1:3);
r_G=LAMBDA(n_grid:end-2);
G_J(2:end-1,2:end-1)=toeplitz(c_G,r_G);
G_J(end-1:-1:2,1)=lambda*p_m*exp(((-n_grid+2:-1)+1/2)*h*eta_m);
G_J(end-1:-1:2,end)=lambda*p_p*exp(-((1:n_grid-2)-1/2)*h*eta_p);

G=G_D+G_J;
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
%     Prob=Prob+real(J_q(n_half+1,:,n_half+1)./q)*coe_inv;
% end
% 
% Prob

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
    



 
% V1=0.802678454549736;V2=0.812342736348707;V3=0.817375561134044;V4=0.819942603926821;V5=0.821238832615252;V6=0.821890129496243;
% 1/21*(64*V5-56*V4+14*V3-V2)
% 1/315*(1024*V5-960*V4+280*V3-30*V2+V1)
 

 
 
toc;

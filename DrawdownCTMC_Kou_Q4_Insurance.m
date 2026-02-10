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
 
% a=0.2;  % drawdown level
alpha=0.25;
a=-log(1-alpha);

% n_drawdown=8;

%% CTMC approximation
n_a=160;
h=a/n_a;
y_CTMC=(-n_a-1:1)*h;
n_grid=length(y_CTMC);


% Construction of transition rate matrix
sigma_bar2=lambda*(p_p*(2/eta_p^2-(1/4*h^2+1/eta_p*h+2/eta_p^2)*exp(-1/2*eta_p*h))+p_m*(2/eta_m^2-(1/4*h^2+1/eta_m*h+2/eta_m^2)*exp(-1/2*eta_m*h)));
LAMBDA=zeros(2*n_grid-1,1);
LAMBDA(1:n_grid-1)=lambda*p_m*(exp(((-n_grid+1:-1)+1/2)*h*eta_m)-exp(((-n_grid+1:-1)-1/2)*h*eta_m));
LAMBDA(n_grid+1:end)=lambda*p_p*(exp(-((1:n_grid-1)-1/2)*h*eta_p)-exp(-((1:n_grid-1)+1/2)*h*eta_p));

k=floor(1/h-1/2);  % number of intervals within [1/2h,1]
LAMBDA_k=[lambda*p_m*(exp(((-k:-1)+1/2)*h*eta_m)-exp(((-k:-1)-1/2)*h*eta_m)),0,lambda*p_p*(exp(-((1:k)-1/2)*h*eta_p)-exp(-((1:k)+1/2)*h*eta_p))];
mu_bar=(-k:k)*h*LAMBDA_k.'+(k+1)*h*lambda*(p_p*(exp(-(k+1-1/2)*h*eta_p)-exp(-eta_p))-p_m*(exp(-(k+1-1/2)*h*eta_m)-exp(-eta_m)));
mu=r-d-sigma^2/2-lambda*(p_p*eta_p/(eta_p-1)+p_m*eta_m/(eta_m+1)-1);

c_G=[0;-(mu-mu_bar)/(2*h)+(sigma^2+sigma_bar2)/(2*h^2)+LAMBDA(n_grid-1);LAMBDA(n_grid-2:-1:1)];  % first column of G, G is a Toeplitz matrix
r_G=[0;(mu-mu_bar)/(2*h)+(sigma^2+sigma_bar2)/(2*h^2)+LAMBDA(n_grid+1);LAMBDA(n_grid+2:end)];  % first row of G
G=toeplitz(c_G,r_G);
G(2:end,1)=lambda*p_m*exp(((-1:-1:-n_grid+1)+1/2)*h*eta_m);
G(1:end-1,end)=lambda*p_p*exp(-((n_grid-1:-1:1)-1/2)*h*eta_p);
G=G-diag(sum(G,2));
G(1,:)=zeros(1,n_grid);
G(end,:)=zeros(1,n_grid);
G(1)=-(mu-mu_bar)/(2*h)-(sigma^2+sigma_bar2)/(h^2);
G(1,2)=-G(1);
G(end)=G(1);
G(end,end-1)=-G(end);

 
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

B_m=-sum(G(3:end-1,1:2),2);
B_p=-sum(G(3:end-1,end),2);
Q_a=zeros(1,n_q);
for i=1:n_q
    A_G=G(3:end-1,3:end-1)-(q(i)+r)*eye(n_a);
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
%             A_G=G(1:k,1:k)-(q(i)+r)*eye(k);
%             P_S=A_G\B_p(:,i);
%             h_q(k,i)=P_S(end);
%         end
%     end
%     h_q_last=h_q;
%     Prob=Prob+real(h_q(n_half+1,:)./q)*coe_inv;
% end
%  
% Prob

% V1=1.222091415938574;V2=1.249719656018007;V3=1.264502665712635;V4=1.272145954453273;V5=1.276031768524759;V6=1.277990872759138;
% 1/21*(64*V4-56*V3+14*V2-V1)
% 1/315*(1024*V6-960*V5+280*V4-30*V3+V2)

 

 


 
 
toc;

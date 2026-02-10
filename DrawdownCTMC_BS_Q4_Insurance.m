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
 
% BS model
sigma=0.3;  % volatility
 
alpha=0.25;
a=-log(1-alpha);  % drawdown level
 
n_drawdown=6;

%% CTMC approximatio
n_a=160;
h=a/n_a;
% upper_bound=1.2;
% n_half=ceil(upper_bound/h);
% y_CTMC=(-n_half:n_half)*h;
% n_grid=length(y_CTMC);
y_CTMC=(-n_a:1)*h;
n_grid=length(y_CTMC);
 
 
 
% Construction of transition rate matrix
G_0=[0,-sigma^2/h^2*ones(1,n_grid-2),0];
G_1=[0,(r-d-sigma^2/2)/(2*h)*ones(1,n_grid-2)+sigma^2/(2*h^2)*ones(1,n_grid-2)];
G_2=[-(r-d-sigma^2/2)/(2*h)*ones(1,n_grid-2)+sigma^2/(2*h^2)*ones(1,n_grid-2),0];
G=diag(G_0)+diag(G_1,1)+diag(G_2,-1);
 
% p=expm(G*T);  % transition probability
 
%% Laplace transform
A=15;k_1=10;k_2=10;
q=A/(2*T)+(0:k_1+k_2)*pi*1i/T;  % Laplace inversion grid
n_q=length(q);
 
Psi_p=zeros(n_grid,1,n_q);
Psi_m=zeros(n_grid,1,n_q);
Psi_p(end,1,:)=1;
Psi_m(1,1,:)=1;
B_p=zeros(n_grid-2,1);
B_m=zeros(n_grid-2,1);
B_p(end)=-G(end-1,end);
B_m(1)=-G(2,1);
for i=1:n_q
    A_G=G(2:end-1,2:end-1)-(q(i)+r)*eye(n_grid-2);
    Psi_p(2:end-1,1,i)=A_G\B_p;
    Psi_m(2:end-1,1,i)=A_G\B_m;
end
 


% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;



P_left=(Psi_p(end,:).*Psi_m(end-1,:)-Psi_p(end-1,:).*Psi_m(end,:))./...
    (Psi_p(end,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(end,:));
P_right=(Psi_p(end-1,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(end-1,:))./...
    (Psi_p(end,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(end,:));
h_q=P_left./(1-P_right);
h_a=h_q./(1-h_q);
Prob=real(h_a./q)*coe_inv

% % Recursion
% Prob=0;
% h_q=ones(1,n_q);
% for n=1:n_drawdown
%     P_left=(Psi_p(end,:).*Psi_m(end-1,:)-Psi_p(end-1,:).*Psi_m(end,:))./...
%         (Psi_p(end,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(end,:));
%     P_right=(Psi_p(end-1,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(end-1,:))./...
%         (Psi_p(end,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(end,:));
%     h_q=P_left.*h_q./(1-P_right);
% 
%     Prob=Prob+real(h_q./q)*coe_inv;
% end
%  
% Prob

% V1=0.874178675235495;V2=0.898642712833248;V3=0.911484814394377;V4=0.918065001190654;V5=0.921395744118688;V6=0.923071388692062;
% 1/21*(64*V6-56*V5+14*V4-V3)
 

 

 
 
toc;

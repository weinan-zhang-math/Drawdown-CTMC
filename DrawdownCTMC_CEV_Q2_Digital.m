clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
S_0=1;
Y_0=log(S_0);  % initial asset price
 
T=0.5;  % maturity
 
% CEV model
sigma=0.3;  % volatility
beta=-0.25;
 
a=0.2;  % drawdown level
 
xi=0.1; 
 
%% CTMC approximation
n_a=160;
h=a/n_a;
% n_0=ceil(Y_0/h);
upper_bound=3;
n_half=ceil(upper_bound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=2*n_half+1;
 
 
 
% Construction of transition rate matrix
% G_0=[0,-sigma^2*y_CTMC(2:end-1).^(2+2*beta)/h^2,0];
% G_1=[0,(r-d)*y_CTMC(2:end-1)/(2*h)+sigma^2*y_CTMC(2:end-1).^(2+2*beta)/(2*h^2)];
% G_2=[-(r-d)*y_CTMC(2:end-1)/(2*h)+sigma^2*y_CTMC(2:end-1).^(2+2*beta)/(2*h^2),0];
% G=diag(G_0)+diag(G_1,1)+diag(G_2,-1);
 
G_0=[0,-sigma^2*exp(2*beta*y_CTMC(2:end-1))/h^2,0];
G_1=[0,(r-d-1/2*sigma^2*exp(2*beta*y_CTMC(2:end-1)))/(2*h)+sigma^2*exp(2*beta*y_CTMC(2:end-1))/(2*h^2)];
G_2=[-(r-d-1/2*sigma^2*exp(2*beta*y_CTMC(2:end-1)))/(2*h)+sigma^2*exp(2*beta*y_CTMC(2:end-1))/(2*h^2),0];
G=diag(G_0)+diag(G_1,1)+diag(G_2,-1);

% p=expm(G*T);  % transition probability
 
%% Laplace transform
A=15;k_1=10;k_2=10;
q=A/(2*T)+(0:k_1+k_2)*pi*1i/T;  % Laplace inversion grid
n_q=length(q);
 
Psi_p=zeros(n_grid,n_q);
Psi_m=zeros(n_grid,n_q);
Psi_p(end,:)=1;
Psi_m(1,:)=1;
B_p=zeros(n_grid-2,1);
B_m=zeros(n_grid-2,1);
B_p(end)=-G(end-1,end);
B_m(1)=-G(2,1);

Indicate=(y_CTMC(2:end-1)<xi);

for i=1:n_q
    A_G=G(2:end-1,2:end-1)-diag(q(i)*Indicate+r);
    Psi_p(2:end-1,i)=A_G\B_p;
    Psi_m(2:end-1,i)=A_G\B_m;
end
 
% Recursion
 
 
B_q=zeros(n_grid,n_q);
% B_q(end,:)=1;

B_q(n_grid-1,:)=(Psi_p(n_grid,:).*Psi_m(n_grid-1,:)-Psi_p(n_grid-1,:).*Psi_m(n_grid,:))./...
    (Psi_p(n_grid,:).*Psi_m(n_grid-n_a-1,:)-Psi_p(n_grid-n_a-1,:).*Psi_m(n_grid,:))+...
    (Psi_p(n_grid-1,:).*Psi_m(n_grid-n_a-1,:)-Psi_p(n_grid-n_a-1,:).*Psi_m(n_grid-1,:))./...
    (Psi_p(n_grid,:).*Psi_m(n_grid-n_a-1,:)-Psi_p(n_grid-n_a-1,:).*Psi_m(n_grid,:)).*B_q(end,:);

 
for k=n_grid-2:-1:n_half+1
    P_left=(Psi_p(k+1,:).*Psi_m(k,:)-Psi_p(k,:).*Psi_m(k+1,:))./...
        (Psi_p(k+1,:).*Psi_m(k-n_a,:)-Psi_p(k-n_a,:).*Psi_m(k+1,:));
    P_right=(Psi_p(k,:).*Psi_m(k-n_a,:)-Psi_p(k-n_a,:).*Psi_m(k,:))./...
        (Psi_p(k+1,:).*Psi_m(k-n_a,:)-Psi_p(k-n_a,:).*Psi_m(k+1,:));
    B_q(k,:)=P_left+P_right.*B_q(k+1,:);
end

h_a=B_q(k,:);


 
%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;
 
%% Probability/Digital option

Prob=real(h_a./q)*coe_inv

% V1=0.894954775912391;V2=0.898606997411586;V3=0.900408168206584;V4=0.901301905732562;V5=0.901746987744120;V6=0.901969072816589;
% 1/21*(64*V6-56*V5+14*V4-V3)
 
 
toc;

clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
S_0=1;
Y_0=log(S_0);  % initial asset price
 
T=0.1;  % maturity
 
% CEV model
sigma=0.3;  % volatility
beta=-0.25;
 
a=0.2;  % drawdown level

xi=0.1;  % drawup level

%% CTMC approximation
n_a=160;
h=a/n_a;
% n_0=ceil(Y_0/h);
upper_bound=2.4;
n_half=ceil(upper_bound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=2*n_half+1;

n_xi=floor(xi/h)+1;

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
 
% Recursion
C_q=zeros(n_grid,n_q);

% C_q(end,:)=1;

Indicate=diag([ones(n_a-n_xi,1);zeros(n_xi,1)]);
B_m=zeros(n_a,1);
B_p=zeros(n_a,n_q);

for k=n_grid-1:-1:n_half+1
    B_m(1)=-G(k-n_a+1,k-n_a);
    B_p(end,:)=-G(k,k+1)*C_q(k+1,:);
    % B_m=-G(k-n_a+1:k,1:k-n_a)*ones(k-n_a,1);
    % B_p=-G(k-n_a+1:k,k+1:end)*C_q(k+1:end,:);
    for i=1:n_q
        A_G=G(k-n_a+1:k,k-n_a+1:k)-(q(i)*Indicate+r*eye(n_a));
        P_left=A_G\B_m;
        R_right=A_G\B_p(:,i);
        C_q(k,i)=P_left(end)+R_right(end);
    end
end

h_a=C_q(k,:);
 
%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;
 
%% Probability/Digital option
 
Prob=real(h_a./q)*coe_inv

% V1=0.618260707898760;V2=0.595010738187276;V3=0.583683221472252;V4=0.578101745356719;V5=0.575332492286343;V6=0.573953346622307;
% 1/21*(64*V5-56*V4+14*V3-V2)

toc;

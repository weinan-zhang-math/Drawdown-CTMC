% clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
S_0=1;  % initial asset price
Y_0=log(S_0);  % initial log-price
 
T=0.1;  % maturity
 
% BS model
sigma=0.3;  % volatility
 
a=0.2;  % drawdown level

xi=0.1;  % drawup level

%% CTMC approximation
n_a=160;
h=a/n_a;
upper_bound=0.2;
n_half=ceil(upper_bound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=length(y_CTMC);

n_xi=floor(xi/h)+1;
 
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
 
% Recursion
C_q=zeros(n_grid,n_q);

% C_q(end,:)=1;

Indicate=diag([ones(n_a-n_xi,1);zeros(n_xi,1)]);

B_m=-G(n_half-n_a+2:n_half+1,n_half+1-n_a);
B_p=-G(n_half-n_a+2:n_half+1,n_half+2);
for i=1:n_q
    A_G=G(n_half-n_a+2:n_half+1,n_half-n_a+2:n_half+1)-(q(i)*Indicate+r*eye(n_a));
    P_left=A_G\B_m;
    R_right=A_G\B_p;
    C_q(n_half+1,i)=P_left(end)/(1-R_right(end));
end

h_a=C_q(n_half+1,:);


 
%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;
 
%% Probability/Digital option
 
Prob=real(h_a./q)*coe_inv

% V1=0.624244649308548;V2=0.600569971295499;V3=0.589024476915585;V4=0.583333394555896;V5=0.580509278131384;V6=0.579102697204664;
% 1/21*(64*V6-56*V5+14*V4-V3)

toc;

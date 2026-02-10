clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
S_0=1;
S_max=S_0;
S_min=0.9;
Y_0=log(S_0);  % initial log-price
Y_max=Y_0;
Y_min=log(S_min);
 
T=0.5;  % maturity
 
% CEV model
sigma=0.3;  % volatility
beta=-0.25;

a=0.2;  % drawdown level
 
b=0.3;  % drawup level
 
%% CTMC approximation
n_a=160;
h=a/n_a;
n_b=ceil(b/h);
% n_0=ceil(Y_0/h);
% upper_bound=1.3;
% n_half=ceil((upper_bound-Y_0)/h);
% y_CTMC=(-n_half:n_half)*h+Y_0;
% n_grid=length(y_CTMC);
upper_bound=0.3;
n_half=ceil(upper_bound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=length(y_CTMC);

% Construction of transition rate matrix
% G_0=[0,-sigma^2*y_CTMC(2:end-1).^(2+2*beta)/h^2,0];
% G_1=[0,(r-d)*y_CTMC(2:end-1)/(2*h)+sigma^2*y_CTMC(2:end-1).^(2+2*beta)/(2*h^2)];
% G_2=[-(r-d)*y_CTMC(2:end-1)/(2*h)+sigma^2*y_CTMC(2:end-1).^(2+2*beta)/(2*h^2),0];
% G=diag(G_0)+diag(G_1,1)+diag(G_2,-1);

G_0=[0,-sigma^2*exp(2*beta*y_CTMC(2:end-1))/h^2,0];
G_1=[0,(r-d-1/2*sigma^2*exp(2*beta*y_CTMC(2:end-1)))/(2*h)+sigma^2*exp(2*beta*y_CTMC(2:end-1))/(2*h^2)];
G_2=[-(r-d-1/2*sigma^2*exp(2*beta*y_CTMC(2:end-1)))/(2*h)+sigma^2*exp(2*beta*y_CTMC(2:end-1))/(2*h^2),0];
G=diag(G_0)+diag(G_1,1)+diag(G_2,-1);
% p=expm(G*T); 
 
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
    A_G=G(2:end-1,2:end-1)-q(i)*eye(n_grid-2);
    Psi_p(2:end-1,1,i)=A_G\B_p;
    Psi_m(2:end-1,1,i)=A_G\B_m;
end

% Recursion

A_q=zeros(n_grid,n_grid,n_q);
for k=n_grid-1:-1:n_half+1
    P_left=(Psi_p(k+1,1,:).*Psi_m(k,1,:)-Psi_p(k,1,:).*Psi_m(k+1,1,:))./...
        (Psi_p(k+1,1,:).*Psi_m(k-n_a,1,:)-Psi_p(k-n_a,1,:).*Psi_m(k+1,1,:));
    R_right=(Psi_p(k,1,:).*Psi_m(k-n_a,1,:)-Psi_p(k-n_a,1,:).*Psi_m(k,1,:))./...
        (Psi_p(k+1,1,:).*Psi_m(k-n_a,1,:)-Psi_p(k-n_a,1,:).*Psi_m(k+1,1,:));
    A_q(k,k-n_b+1:k-n_a+1,:)=P_left+R_right.*A_q(k+1,k-n_b+1:k-n_a+1,:);
    R_diag=(Psi_p(k-n_a+1,1,:).*Psi_m(k-n_a,1,:)-Psi_p(k-n_a,1,:).*Psi_m(k-n_a+1,1,:))./...
        (Psi_p(k+1,1,:).*Psi_m(k-n_a,1,:)-Psi_p(k-n_a,1,:).*Psi_m(k+1,1,:)).*A_q(k+1,k-n_a+1,:);
    for k1=k-n_a+2:k
        R_right1=(Psi_p(k+1,1,:).*Psi_m(k,1,:)-Psi_p(k,1,:).*Psi_m(k+1,1,:))./...
            (Psi_p(k+1,1,:).*Psi_m(k1-1,1,:)-Psi_p(k1-1,1,:).*Psi_m(k+1,1,:));
        R_right2=(Psi_p(k,1,:).*Psi_m(k1-1,1,:)-Psi_p(k1-1,1,:).*Psi_m(k,1,:))./...
            (Psi_p(k+1,1,:).*Psi_m(k1-1,1,:)-Psi_p(k1-1,1,:).*Psi_m(k+1,1,:));
        A_q(k,k1,:)=P_left+R_right1.*R_diag+R_right2.*A_q(k+1,k1,:);
        R_diag1=(Psi_p(k+1,1,:).*Psi_m(k1,1,:)-Psi_p(k1,1,:).*Psi_m(k+1,1,:))./...
            (Psi_p(k+1,1,:).*Psi_m(k1-1,1,:)-Psi_p(k1-1,1,:).*Psi_m(k+1,1,:));
        R_diag2=(Psi_p(k1,1,:).*Psi_m(k1-1,1,:)-Psi_p(k1-1,1,:).*Psi_m(k1,1,:))./...
            (Psi_p(k+1,1,:).*Psi_m(k1-1,1,:)-Psi_p(k1-1,1,:).*Psi_m(k+1,1,:));
        R_diag=R_diag1.*R_diag+R_diag2.*A_q(k+1,k1,:);
    end
end
    
n_left=ceil(abs(Y_min-Y_0)/h);
n_right=n_left-1;
h_a_left=A_q(k,k-n_left,:);
h_a_right=A_q(k,k-n_right,:);
h_a_left=reshape(h_a_left,1,n_q);
h_a_right=reshape(h_a_right,1,n_q);





 
%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;
 
%% Probability/Digital option
 
Prob_left=real(h_a_left./q)*coe_inv;
Prob_right=real(h_a_right./q)*coe_inv;
% Prob=(Y_min-(n_0+1-n_left)*h)/h*Prob_right+((n_0+1-n_right)*h-Y_min)/h*Prob_left
Prob=(n_left*h+Y_min)/h*Prob_right+(-n_right*h-Y_min)/h*Prob_left


% V1=0.511710058194846;V2=0.525605340460836;V3=0.532702223561768;V4=0.536288235841626;V5=0.538090673219769;V6=0.538994254958268;
% 1/21*(64*V4-56*V3+14*V2-V1)

% V1=0.551519842962995;V2=0.559286115366785;V3=0.563096570241454;V4=0.565000659326547;V5=0.565952375448012;V6=0.566427941382503;
% 1/21*(64*V6-56*V5+14*V4-V3)

toc;

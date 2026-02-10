clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
Y_0=0;  % initial asset price
 
T=1;  % maturity
 
% CEV model
sigma=0.3;  % volatility
beta=-0.25;
 
% a=0.2;  % drawdown level
alpha=0.25;
a=-log(1-alpha);

n_drawdown=8;

%% CTMC approximation
n_a=160;
h=a/n_a;
upper_bound=2;
n_half=ceil(upper_bound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=2*n_half+1;
 
 
 
% Construction of transition rate matrix
G_0=[0,-sigma^2*exp(2*beta*y_CTMC(2:end-1))/h^2,0];
G_1=[0,(r-d-1/2*sigma^2*exp(2*beta*y_CTMC(2:end-1)))/(2*h)+sigma^2*exp(2*beta*y_CTMC(2:end-1))/(2*h^2)];
G_2=[-(r-d-1/2*sigma^2*exp(2*beta*y_CTMC(2:end-1)))/(2*h)+sigma^2*exp(2*beta*y_CTMC(2:end-1))/(2*h^2),0];
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
h_q=zeros(n_grid,n_q);
tic
for i=1:n_q
    A_G=G(2:end-1,2:end-1)-((q(i)+r)*eye(n_grid-2));
    Psi_p(2:end-1,1,i)=A_G\B_p;
    Psi_m(2:end-1,1,i)=A_G\B_m;
    
    diag_H=ones(n_grid-2,1);
    diag_H_up=(Psi_p(2:n_grid-2,i).*Psi_m(max((2:n_grid-2)-n_a,1),i)-Psi_p(max((2:n_grid-2)-n_a,1),i).*Psi_m(2:n_grid-2,i))./...
        (Psi_p(3:n_grid-1,i).*Psi_m(max((2:n_grid-2)-n_a,1),i)-Psi_p(max((2:n_grid-2)-n_a,1),i).*Psi_m(3:n_grid-1,i));
    diag_H_down=(Psi_p(n_a+3:end,i).*Psi_m(n_a+2:end-1,i)-Psi_p(n_a+2:end-1,i).*Psi_m(n_a+3:end,i))./...
        (Psi_p(n_a+3:end,i).*Psi_m(2:end-n_a-1,i)-Psi_p(2:end-n_a-1,i).*Psi_m(n_a+3:end,i));
    A_H=diag(diag_H)-diag(diag_H_up,1)-diag(diag_H_down,-n_a);
    A_H=sparse(A_H);
    B_H=[zeros(n_a,1);diag_H_down];
    h_q(2:end-1,i)=A_H\B_H;
end
% toc
%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';

coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;

Prob=real(h_q(n_half+1,:)./q)*coe_inv

% Recursion
% h_q=zeros(n_grid,n_q);
% h_q_last=ones(n_grid,n_q);
% Prob=0;
% for n=1:n_drawdown
%     for k=n_grid-1:-1:n_a+1
%         P_left=(Psi_p(k+1,:).*Psi_m(k,:)-Psi_p(k,:).*Psi_m(k+1,:))./...
%             (Psi_p(k+1,:).*Psi_m(k-n_a,:)-Psi_p(k-n_a,:).*Psi_m(k+1,:));
%         P_right=(Psi_p(k,:).*Psi_m(k-n_a,:)-Psi_p(k-n_a,:).*Psi_m(k,:))./...
%             (Psi_p(k+1,:).*Psi_m(k-n_a,:)-Psi_p(k-n_a,:).*Psi_m(k+1,:));
%         h_q(k,:)=P_left.*h_q_last(k-n_a,:)+P_right.*h_q(k+1,:);
%     end
%     for k=n_a:-1:1
%         P_right=(Psi_p(k,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(k,:))./...
%             (Psi_p(k+1,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(k+1,:));
%         h_q(k,:)=P_right.*h_q(k+1,:);
%     end
%     h_q_last=h_q;
%     Prob=Prob+real(h_q(n_half+1,:)./q)*coe_inv;
% end
% 
% Prob

% V1=0.893018939891304;V2=0.917679943884759;V3=0.930618032316807;V4=0.937245592492350;V5=0.940599867629888;V6=0.942287240142467;
% 1/21*(64*V5-56*V4+14*V3-V2)
 

 
toc;

% clear;
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
 
n_drawdown=4;
 
%% CTMC approximation
n_a=160;
h=a/n_a;
upper_bound=0.8;
n_half=ceil(upper_bound/h);
y_CTMC=(-n_half:n_half)*h;
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
 
P_left=(Psi_p(n_half+2,:).*Psi_m(n_half+1,:)-Psi_p(n_half+1,:).*Psi_m(n_half+2,:))./...
    (Psi_p(n_half+2,:).*Psi_m(n_half+1-n_a,:)-Psi_p(n_half+1-n_a,:).*Psi_m(n_half+2,:));
P_right=(Psi_p(n_half+1,:).*Psi_m(n_half+1-n_a,:)-Psi_p(n_half+1-n_a,:).*Psi_m(n_half+1,:))./...
    (Psi_p(n_half+2,:).*Psi_m(n_half+1-n_a,:)-Psi_p(n_half+1-n_a,:).*Psi_m(n_half+2,:));
P_right_inf=(Psi_p(n_half+1-n_a,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(n_half+1-n_a,:))./...
    (Psi_p(n_half+1,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(n_half+1,:));

J_q_diag=P_left./(1-P_right-P_left.*P_right_inf);




% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;

Prob=real(J_q_diag./q)*coe_inv

% % Recursion
% J_q=zeros(n_grid,n_q,n_grid);
% J_q_last=ones(n_grid,n_q,n_grid);
% J_q_diag=zeros(n_grid,n_q);
% Prob=0;
% for n=1:n_drawdown
%     for k_y=n_grid-1:-1:n_a+1
%         P_diag_right=(Psi_p(k_y,:).*Psi_m(k_y-n_a,:)-Psi_p(k_y-n_a,:).*Psi_m(k_y,:))./...
%             (Psi_p(k_y+1,:).*Psi_m(k_y-n_a,:)-Psi_p(k_y-n_a,:).*Psi_m(k_y+1,:));
%         P_diag_left=(Psi_p(k_y+1,:).*Psi_m(k_y,:)-Psi_p(k_y,:).*Psi_m(k_y+1,:))./...
%             (Psi_p(k_y+1,:).*Psi_m(k_y-n_a,:)-Psi_p(k_y-n_a,:).*Psi_m(k_y+1,:));
%         J_q_diag(k_y,:)=P_diag_right.*J_q_diag(k_y+1,:)+P_diag_left.*J_q_last(k_y-n_a,:,k_y);
% 
%         J_q(2:k_y-1,:,k_y)=(Psi_p(2:k_y-1,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(2:k_y-1,:))./...
%             (Psi_p(k_y,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(k_y,:)).*J_q_diag(k_y,:);
%         J_q(k_y,:,k_y)=J_q_diag(k_y,:);
%     end
%     for k_y=n_a:-1:1
%         P_diag_right=(Psi_p(k_y,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(k_y,:))./...
%             (Psi_p(k_y+1,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(k_y+1,:));
%         J_q_diag(k_y,:)=P_diag_right.*J_q_diag(k_y+1,:);
% 
%         J_q(2:k_y-1,:,k_y)=(Psi_p(2:k_y-1,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(2:k_y-1,:))./...
%             (Psi_p(k_y,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(k_y,:)).*J_q_diag(k_y,:);
%         J_q(k_y,:,k_y)=J_q_diag(k_y,:);
%     end
%     J_q_last=J_q;
% 
%     h_a=J_q(n_half+1,:,n_half+1);
%     Prob=Prob+real(h_a./q)*coe_inv;
% end
% 
% Prob

% V1=0.656790533484852;V2=0.668311157092377;V3=0.674186405275467;V4=0.677152693653601;V5=0.678642995407950;V6=0.679389934609834;
% 1/21*(64*V4-56*V3+14*V2-V1)
% 1/315*(1024*V6-960*V5+280*V4-30*V3+V2)

toc;

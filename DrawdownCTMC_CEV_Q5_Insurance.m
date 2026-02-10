clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
Y_0=1;  % initial asset price
 
T=1;  % maturity
 
% CEV model
sigma=0.3;  % volatility
beta=-0.25;
 
% a=0.1;  % drawdown level
alpha=0.25;
a=-log(1-alpha);

n_drawdown=3;
 
%% CTMC approximation
n_a=160;
h=a/n_a;
upper_bound=1.2;
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
for i=1:n_q
    A_G=G(2:end-1,2:end-1)-(q(i)+r)*eye(n_grid-2);
    Psi_p(2:end-1,1,i)=A_G\B_p;
    Psi_m(2:end-1,1,i)=A_G\B_m;
end

% Recursion (simplified)
J_q_diag=zeros(n_grid,n_q);
for k=n_grid-1:-1:n_half+1
    P_num_right=(Psi_p(k,:).*Psi_m(k-n_a,:)-Psi_p(k-n_a,:).*Psi_m(k,:))./...
        (Psi_p(k+1,:).*Psi_m(k-n_a,:)-Psi_p(k-n_a,:).*Psi_m(k+1,:));
    P_num_left=(Psi_p(k+1,:).*Psi_m(k,:)-Psi_p(k,:).*Psi_m(k+1,:))./...
        (Psi_p(k+1,:).*Psi_m(k-n_a,:)-Psi_p(k-n_a,:).*Psi_m(k+1,:));
    
    P_den_left=P_num_left;
    P_den_right=(Psi_p(k-n_a,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(k-n_a,:))./...
        (Psi_p(k,:).*Psi_m(1,:)-Psi_p(1,:).*Psi_m(k,:));
    
    J_q_diag(k,:)=(P_num_right.*J_q_diag(k+1,:)+P_num_left)./(1-P_den_left.*P_den_right);
end

 
%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;

Prob=real(J_q_diag(n_half+1,:)./q)*coe_inv

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
%     Prob=Prob+real(J_q(n_half+1,:,n_half+1)./q)*coe_inv;
% end
% 
% Prob



 
 
% V1=0.637016479208910;V2=0.647925585969933;V3=0.653496720348974;V4=0.656311518650219;V5=0.657726236342892;V6=0.658435426328124;
% 1/21*(64*V6-56*V5+14*V4-V3)
% 1/315*(1024*V6-960*V5+280*V4-30*V3+V2)
 
toc;

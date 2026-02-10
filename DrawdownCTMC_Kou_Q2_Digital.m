clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
S_0=1;  % initial asset price
Y_0=log(S_0);  % initial log-price

T=0.5;

% Kou model
sigma=0.3;  % volatility
p_p=0.5;
p_m=1-p_p;
eta_p=10;
eta_m=10;
lambda=3;

a=0.2;  % drawdown level
 
xi=0.1;  % drawup level
%% CTMC approximation
n_a=160;
h=a/n_a;
upperbound=3;
n_half=ceil(upperbound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=2*n_half+1;


% Construction of transition rate matrix
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


% p=expm(G*T);  % transition probability
 
%% Laplace transform
A=15;k_1=10;k_2=10;
q=A/(2*T)+(0:k_1+k_2)*pi*1i/T;  % Laplace inversion grid
n_q=length(q);


% Recursion

% % B_q=zeros(n_grid,n_q);
% % Index_xi_a=find(y_CTMC-a+h<=xi,1,'last');
% % B_q(end,:)=1;
% % Indicate=y_CTMC<xi;
% 
% for i=1:n_q
%     for k=n_grid-1:-1:n_half+1
%         B_m=-sum(G(k-n_a+1:k,1:k-n_a),2);
%         B_p=-G(k-n_a+1:k,k+1:end)*B_q(k+1:end,i);
%         A_G=G(k-n_a+1:k,k-n_a+1:k)-diag(q(i)*Indicate(k-n_a+1:k)+r);
%         P_S=A_G\(B_m+B_p);
%         B_q(k,i)=P_S(end);
%     end
% 
% %     for k=Index_xi_a-1:-1:max(Index_xi_a-n_a,n_half+1)
% %         B_m=-sum(G(k-n_a+1:k,1:k-n_a),2);
% %         B_p=-G(k-n_a+1:k,k+1:end)*B_q(k+1:end,i);
% %         A_G=G(k-n_a+1:k,k-n_a+1:k)-diag(q(i)*Indicate(k-n_a+1:k)+r);
% %         if k==Index_xi_a-1
% %             A_G_inverse=A_G\eye(n_a);
% %             A_G_last=A_G;
% %             A_G_inverse_last=A_G_inverse;
% %         else
% %             B_G=A_G-A_G_last;
% %             A_G_inverse=A_G_inverse_last-A_G_inverse_last*B_G*A_G_inverse_last/(1+trace(B_G*A_G_inverse_last));
% %             A_G_last=A_G;
% %             A_G_inverse_last=A_G_inverse;
% %         end
% %         P_S=A_G_inverse*(B_m+B_p);
% %         B_q(k,i)=P_S(end);
% %     end
% %     
% %     for k=max(Index_xi_a-n_a,n_half+1)-1:-1:n_half+1
% %         B_m=-sum(G(k-n_a+1:k,1:k-n_a),2);
% %         B_p=-G(k-n_a+1:k,k+1:end)*B_q(k+1:end,i);
% %         A_G=G(k-n_a+1:k,k-n_a+1:k)-diag(q(i)*Indicate(k-n_a+1:k)+r);
% %         P_S=A_G\(B_m+B_p);
% %         B_q(k,i)=P_S(end);
% %     end
% end



B_q=zeros(n_grid,n_q);

% B_q(end,:)=1;
Indicate=y_CTMC<xi;

for k=n_grid-1:-1:n_half+1
    B_m=-G(k-n_a+1:k,1:k-n_a)*ones(k-n_a,1);
    B_p=-G(k-n_a+1:k,k+1:end)*B_q(k+1:end,:);
    for i=1:n_q
        A_G=G(k-n_a+1:k,k-n_a+1:k)-diag(q(i)*Indicate(k-n_a+1:k)+r);
        P_S=A_G\(B_m+B_p(:,i));
        B_q(k,i)=P_S(end);
    end
end

h_a=B_q(n_half+1,:);




%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;
 
%% Probability/Digital option
 
Prob=real(h_a./q)*coe_inv

% V1=0.937428986873779;V2=0.939782509084369;V3=0.940952419591790;V4=0.941535159230894;V5=0.941825913036375;V6=0.941971128221970;
% 1/21*(64*V5-56*V4+14*V3-V2) 

toc;

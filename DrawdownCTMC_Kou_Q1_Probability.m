clear;
format long;
tic;
%% Parameter
r=0.05;  % interest
d=0.02;  % dividend
S_0=1;  % initial asset price
S_max=S_0;
S_min=0.9;
Y_0=log(S_0);  % initial log-price
Y_max=Y_0;
Y_min=log(S_min);
 
T=0.5;  % maturity
 
% Kou model
sigma=0.3;  % volatility
p_p=0.5;
p_m=1-p_p;
eta_p=10;
eta_m=10;
lambda=3;

a=0.2;  % drawdown level
 
b=0.3;  % drawup level

%% CTMC approximation
n_a=160;
h=a/n_a;
n_b=ceil(b/h);
upper_bound=0.3;
n_half=ceil(upper_bound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=2*n_half+1;

 
 
% % Construction of transition rate matrix
% sigma_bar2=lambda*(p_p*(2/eta_p^2-(1/4*h^2+1/eta_p*h+2/eta_p^2)*exp(-1/2*eta_p*h))+p_m*(2/eta_m^2-(1/4*h^2+1/eta_m*h+2/eta_m^2)*exp(-1/2*eta_m*h)));
% LAMBDA=zeros(2*n_grid-1,1);
% LAMBDA(1:n_grid-1)=lambda*p_m*(exp(((-n_grid+1:-1)+1/2)*h*eta_m)-exp(((-n_grid+1:-1)-1/2)*h*eta_m));
% LAMBDA(n_grid+1:end)=lambda*p_p*(exp(-((1:n_grid-1)-1/2)*h*eta_p)-exp(-((1:n_grid-1)+1/2)*h*eta_p));
% 
% k=floor(1/h-1/2);  % number of intervals within [1/2h,1]
% LAMBDA_k=[lambda*p_m*(exp(((-k:-1)+1/2)*h*eta_m)-exp(((-k:-1)-1/2)*h*eta_m)),0,lambda*p_p*(exp(-((1:k)-1/2)*h*eta_p)-exp(-((1:k)+1/2)*h*eta_p))];
% mu_bar=(-k:k)*h*LAMBDA_k.'+(k+1)*h*lambda*(p_p*(exp(-(k+1-1/2)*h*eta_p)-exp(-eta_p))-p_m*(exp(-(k+1-1/2)*h*eta_m)-exp(-eta_m)));
% 
% 
% mu=r-d-sigma^2/2-lambda*(p_p*eta_p/(eta_p-1)+p_m*eta_m/(eta_m+1)-1);
% 
% c_G=[0;-(mu-mu_bar)/(2*h)+(sigma^2+sigma_bar2)/(2*h^2)+LAMBDA(n_grid-1);LAMBDA(n_grid-2:-1:1)];  % first column of G, G is a Toeplitz matrix
% r_G=[0;(mu-mu_bar)/(2*h)+(sigma^2+sigma_bar2)/(2*h^2)+LAMBDA(n_grid+1);LAMBDA(n_grid+2:end)];  % first row of G
% G=toeplitz(c_G,r_G);
% G=G-diag(sum(G,2));
% G(1,:)=zeros(1,n_grid);
% G(end,:)=zeros(1,n_grid);
% % p=expm(G*T);  % transition probability

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

 
%% Laplace transform
A=15;k_1=10;k_2=k_1;
q=A/(2*T)+(0:k_1+k_2)*pi*1i/T;  % Laplace inversion grid
n_q=length(q);


% Recursion
q_reshape=reshape(q,1,1,[]);
A_G_q=G(n_grid-n_a:n_grid-1,n_grid-n_a:n_grid-1)-q_reshape.*eye(n_a);
A_k_inverse=cell(n_a,n_q);
for k=1:n_a
    for i=1:n_q
        A_k_inverse{k,i}=A_G_q(1:k,1:k,i)\eye(k);
    end
end

A_q=zeros(n_grid,n_grid,n_q);

for k=n_grid-1:-1:n_grid-n_a
    B_m=-sum(G(k-n_a+1:k,1:k-n_a),2);
%     B_m=-G(k-n_a+1:k,1:k-n_a)*max(K_strike-exp(y_CTMC(1:k-n_a).'),0);
    for i=1:n_q
        B_p=-G(k-n_a+1:k,k+1:min(k-n_a+n_b,n_grid))*A_q(k+1:min(k-n_a+n_b,n_grid),k-n_b+1:k-n_a+1,i);
        P_left=A_k_inverse{n_a,i}*B_m;
        R_right=A_k_inverse{n_a,i}*B_p;
        A_q(k,k-n_b+1:k-n_a+1,i)=P_left(end)+R_right(end,:);
        R_k=R_right(1,end);
        for k1=k-n_a+2:k
            B_k=-G(k1:k,k-n_a+1:k1-1)*R_k-G(k1:k,k+1:min(k1+n_b-1,n_grid))*A_q(k+1:min(k1+n_b-1,n_grid),k1,i);
            R_temp=A_k_inverse{k-k1+1,i}*B_k;
            R_k=[R_k;R_temp(1)];
            A_q(k,k1,i)=P_left(end)+R_temp(end);
        end
    end
end


n_left=ceil(abs(Y_min)/h);
n_right=n_left-1;
h_left=A_q(k,k-n_left,:);
h_right=A_q(k,k-n_right,:);
h_left=reshape(h_left,1,n_q);
h_right=reshape(h_right,1,n_q);
%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';
 
coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in.','reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).'.*coe_inv_in;
 
%% Probability/Digital option
 
% Prob_left=real(h_a_left./q)*coe_inv;
% Prob_right=real(h_a_right./q)*coe_inv;
% Prob=(n_left*h+Y_min)/h*Prob_right+(-n_right*h-Y_min)/h*Prob_left

P_left=real(h_left./q)*coe_inv;
P_right=real(h_right./q)*coe_inv;
P=(n_left*h+Y_min)/h*P_right+(-n_right*h-Y_min)/h*P_left

% Prob=gather(Prob)
 
% V1=0.620491333557921;V2=0.624276399719622;V3=0.626138347210775;V4=0.627080554163974;V5=0.627554533831690;V6=0.627792038538589;
% 1/21*(64*V4-56*V3+14*V2-V1) 

% V1=0.584828063845251;V2=0.590827712587450;V3=0.593777133613025;V4=0.595255922633385;V5=0.595996327107116;V6=0.627792038538589;


toc;


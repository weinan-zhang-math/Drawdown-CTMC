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
 
% Regime Switching BS model
sigma=[0.2 0.4];

Lambda=[-0.75, 0.75; 0.25, -0.25];

a=0.2;  % drawdown level

xi=0.1;


%% CTMC approximation
n_a=640;
h=a/n_a;
upper_bound=2;
n_half=ceil(upper_bound/h);
y_CTMC=(-n_half:n_half)*h;
n_grid=2*n_half+1;

 
n_v=2;
 
% Construction of transition rate matrix
G_Y=zeros(n_grid,n_grid,n_v);

for k_v=1:n_v
    G_Y_0=[0,-sigma(k_v)^2/h^2*ones(1,n_grid-2),0];
    G_Y_1=[0,(r-d-sigma(k_v)^2/2)/(2*h)*ones(1,n_grid-2)+sigma(k_v)^2/(2*h^2)*ones(1,n_grid-2)];
    G_Y_2=[-(r-d-sigma(k_v)^2/2)/(2*h)*ones(1,n_grid-2)+sigma(k_v)^2/(2*h^2)*ones(1,n_grid-2),0];
    G_Y(:,:,k_v)=diag(G_Y_0)+diag(G_Y_1,1)+diag(G_Y_2,-1);
end

% G=kron(G_v,eye(n_grid))+G_Y;
G_Y_1=G_Y(:,:,1);
G_Y_2=G_Y(:,:,2);


%% recursion
A=15;k_1=10;k_2=10;
q=A/(2*T)+(0:k_1+k_2)*pi*1i/T;  % Laplace inversion grid
% q=r;
n_q=length(q);

h=zeros(n_q,n_grid,n_v);

for k_q=1:n_q
    for k_grid=n_grid-1:-1:n_half+1
        Indicate=y_CTMC(k_grid-n_a+1:k_grid)<xi;
        A_G=zeros(n_a*n_v,n_a*n_v);
        A_G(1:n_a,1:n_a)=G_Y(k_grid-n_a+1:k_grid,k_grid-n_a+1:k_grid,1)-diag(q(k_q)*Indicate+r);
        A_G(n_a+1:2*n_a,n_a+1:2*n_a)=G_Y(k_grid-n_a+1:k_grid,k_grid-n_a+1:k_grid,2)-diag(q(k_q)*Indicate+r);
        Lambda_G=kron(Lambda,eye(n_a));

        B_G_right=[-G_Y(k_grid-n_a+1:k_grid,k_grid+1,1),zeros(n_a,1);zeros(n_a,1),-G_Y(k_grid-n_a+1:k_grid,k_grid+1,2)];
        B_G_left=[-G_Y(k_grid-n_a+1:k_grid,k_grid-n_a,1),zeros(n_a,1);zeros(n_a,1),-G_Y(k_grid-n_a+1:k_grid,k_grid-n_a,2)];

        A_Lambda_G=sparse(A_G+Lambda_G);
        S_G_right=A_Lambda_G\B_G_right;
        S_G_left=A_Lambda_G\B_G_left;

        h_q=reshape(h(k_q,k_grid+1,:),1,n_v);
        h(k_q,k_grid,1)=sum(S_G_left(n_a,:))+sum(S_G_right(n_a,:).*h_q);
        h(k_q,k_grid,2)=sum(S_G_left(2*n_a,:))+sum(S_G_right(2*n_a,:).*h_q);
    end
end
    
h_q=reshape(h(:,k_grid,1),1,n_q);

%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';

coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in,'reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).*coe_inv_in;

%% Probability/Digital option


Price=real(h_q./q)*coe_inv

% V1=0.635397393623311;V2=0.639032327782716;V3=0.640965234376862;V4=0.641960585337549;V5=0.642465488958798;V6=0.642719748232781;
% benchmark=0.642975212637817;
% 1/21*(64*V6-56*V5+14*V4-V3)
% 1/315*(1024*V6-960*V5+280*V4-30*V3+V2)


toc;
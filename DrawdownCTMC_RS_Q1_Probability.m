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

K_strike=1;

a=0.2;

b=0.3;
%% CTMC approximation
n_a=640;
h=a/n_a;
n_b=ceil(b/h);
upper_bound=0.5;
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

% g=zeros(n_q,n_grid,n_grid,n_v,n_grid,n_grid,n_v);
h_q=zeros(n_grid,n_grid,n_v,n_q);
% h_diag=zeros(n_q,n_a-1,n_v,n_v);
% h_q=zeros(1,n_q,n_v);
A_Lambda_inv=cell(n_a,n_q);

for k_q=1:n_q
    for k_a=1:n_a
        A_G=zeros(k_a*n_v,k_a*n_v);
        A_G(1:k_a,1:k_a)=G_Y(n_half-k_a+2:n_half+1,n_half-k_a+2:n_half+1,1)-q(k_q)*eye(k_a);
        A_G(k_a+1:end,k_a+1:end)=G_Y(n_half-k_a+2:n_half+1,n_half-k_a+2:n_half+1,2)-q(k_q)*eye(k_a);
        Lambda_G=kron(Lambda,eye(k_a));
        A_Lambda_G=sparse(A_G+Lambda_G);
        A_Lambda_inv(k_a,k_q)={eye(n_v*k_a)/A_Lambda_G};
    end
end


for k_q=1:n_q
    for k_grid=n_grid-1:-1:n_grid-n_a
        % A_G=zeros(n_a*n_v,n_a*n_v);
        % A_G(1:n_a,1:n_a)=G_Y(k_grid-n_a+1:k_grid,k_grid-n_a+1:k_grid,1)-q(k_q)*eye(n_a);
        % A_G(n_a+1:2*n_a,n_a+1:2*n_a)=G_Y(k_grid-n_a+1:k_grid,k_grid-n_a+1:k_grid,2)-q(k_q)*eye(n_a);
        % Lambda_G=kron(Lambda,eye(n_a));
        % A_Lambda_G=sparse(A_G+Lambda_G);

        B_G_left=[-G_Y(k_grid-n_a+1:k_grid,k_grid-n_a,1);-G_Y(k_grid-n_a+1:k_grid,k_grid-n_a,2)];
        S_G_left=A_Lambda_inv{n_a,k_q}*B_G_left;

        B_G_right=[-G_Y(k_grid-n_a+1:k_grid,k_grid+1,1).*h_q(k_grid+1,k_grid-n_b+1:k_grid-n_a+1,1,k_q);...
            -G_Y(k_grid-n_a+1:k_grid,k_grid+1,2).*h_q(k_grid+1,k_grid-n_b+1:k_grid-n_a+1,2,k_q)];
        S_G_right=A_Lambda_inv{n_a,k_q}*B_G_right;

        % h_q=reshape(h(k_q,k_grid+1,k1_grid,:),1,n_v);
        h_q(k_grid,k_grid-n_b+1:k_grid-n_a+1,1,k_q)=S_G_left(n_a,:)+S_G_right(n_a,:);
        h_q(k_grid,k_grid-n_b+1:k_grid-n_a+1,2,k_q)=S_G_left(2*n_a,:)+S_G_right(2*n_a,:);

        
        h_diag=[S_G_right(1,end),S_G_right(n_a+1,end)];
        for k1_grid=k_grid-n_a+2:k_grid
            % A_G=zeros(2*(k_grid-k1_grid+1),2*(k_grid-k1_grid+1));
            % A_G(1:k_grid-k1_grid+1,1:k_grid-k1_grid+1)=...
            %     G_Y(k1_grid:k_grid,k1_grid:k_grid,1)-q(k_q)*eye(k_grid-k1_grid+1);
            % A_G(k_grid-k1_grid+2:2*(k_grid-k1_grid+1),k_grid-k1_grid+2:2*(k_grid-k1_grid+1))=...
            %     G_Y(k1_grid:k_grid,k1_grid:k_grid,2)-q(k_q)*eye(k_grid-k1_grid+1);
            % Lambda_G=kron(Lambda,eye(k_grid-k1_grid+1));
            % A_Lambda_G=sparse(A_G+Lambda_G);

            B_G_right_1=[-G_Y(k1_grid:k_grid,k_grid-n_a+1:k1_grid-1,1)*h_diag(:,1);...
                -G_Y(k1_grid:k_grid,k_grid-n_a+1:k1_grid-1,2)*h_diag(:,2)];
            B_G_right_2=[-G_Y(k1_grid:k_grid,k_grid+1,1).*h_q(k_grid+1,k1_grid,1,k_q);...
                -G_Y(k1_grid:k_grid,k_grid+1,2).*h_q(k_grid+1,k1_grid,2,k_q)];


            S_G_right=A_Lambda_inv{k_grid-k1_grid+1,k_q}*(B_G_right_1+B_G_right_2);

            h_q(k_grid,k1_grid,1,k_q)=S_G_left(n_a,:)+S_G_right(k_grid-k1_grid+1);
            h_q(k_grid,k1_grid,2,k_q)=S_G_left(2*n_a,:)+S_G_right(end);

            h_diag=[h_diag;S_G_right(1),S_G_right(k_grid-k1_grid+2)];
        end
    end
end


n_left=ceil(abs(Y_min)/h);
n_right=n_left-1;
h_q_left=h_q(k_grid,k_grid-n_left,1,:);
h_q_right=h_q(k_grid,k_grid-n_right,1,:);
h_q_left=reshape(h_q_left,1,n_q);
h_q_right=reshape(h_q_right,1,n_q);       
    
    

%% Abate and Whitt Laplace inversion formula
coe_inv=zeros(k_1+k_2+1,1);
coe_inv(1)=exp(A/2)/(2*T);
coe_inv(2:k_1+1)=exp(A/2)/T*(-1).^(1:k_1).';

coe_inv_in=factorial(k_1)./(factorial(1:k_1).*factorial(k_1-1:-1:0));
coe_inv_in=cumsum(coe_inv_in,'reverse')*2^(-k_1);
coe_inv(k_1+2:end)=exp(A/2)/T*(-1).^(k_1+1:k_1+k_2).*coe_inv_in;

%% Probability/Digital option


Price_left=real(h_q_left./q)*coe_inv;
Price_right=real(h_q_right./q)*coe_inv;

Price=(n_left*h+Y_min)/h*Price_right+(-n_right*h-Y_min)/h*Price_left


        


% V1=0.356244250920098;V2=0.364632234944905;V3=0.368923334611157;V4=0.371101896112248;V5=0.372199396178708;
% 1/21*(64*V4-56*V3+14*V2-V1)
% 1/315*(1024*V5-960*V4+280*V3-30*V2+V1)
% benchmark=0.373302134326780;

 
toc;

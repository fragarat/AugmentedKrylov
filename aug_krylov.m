

n = 200; % Size of the matrix
A = zeros(n);

for i = 1:n
    if i <= 4
        A(i, i) = 0.05 * (i/n);
    else
        A(i, i) = i/n;
    end
end

% Generate pseudo-random right-hand side vector
rng(42); % for reproducibility
b = randn(n, 1);

% Initial guess
x0 = zeros(n, 1);

% Standard GMRES without restarts
% gmres(A,b,restart,tol,maxit,M1,M2,x0)
[x1, flag1, relres1, iter1, resvec1] = gmres(A, b, [], 1e-10, 200, [], [], x0);

% GMRES with restarts (Krylov dimension of 40)
[x2, flag2, relres2, iter2, resvec2] = gmres(A, b, 40, 1e-10, 200, [], [], x0);

% Block GMRES without restarts (Block size = 4)
block_size = 4;
[x3, flag3, relres3, iter3, resvec3] = gmres_block(A, b, [], 1e-10, 200, block_size, x0);

% Plot convergence
semilogy(1:length(resvec1), resvec1/norm(b), '-o', 'DisplayName', 'GMRES(inf)');
hold on;
semilogy(1:length(resvec2), resvec2/norm(b), '-s', 'DisplayName', 'GMRES(K=40)');
%semilogy(1:length(resvec3), resvec3/norm(b), '-^', 'DisplayName', 'BGMRES(inf,4)');
hold off;
xlabel('Outer Iterations');
ylabel('Relative Residual Norm');
title('Convergence of different GMRES');
legend('Location', 'best');
grid on;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x, flag, relres, iter, resvec] = gmres_block(A, b, restart, tol, maxit, block_size, x0)
    n = length(b);
    x = x0;
    r = b - A*x;
    beta = norm(r);
    V = zeros(n, block_size);
    H = zeros(block_size+1, block_size);
    flag = 0;
    iter = 0;
    resvec = [];
    
    while beta > tol && iter < maxit
        V(:, 1) = r / beta;
        for i = 1:block_size
            w = A * V(:, i);
            for j = 1:i
                H(j, i) = w' * V(:, j);
                w = w - H(j, i) * V(:, j);
            end
            H(i+1, i) = norm(w);
            if H(i+1, i) < 1e-10
                flag = 1;
                relres = norm(r) / norm(b);
                return;
            end
            V(:, i+1) = w / H(i+1, i);
            % Apply Givens rotations
            for k = 1:i-1
                temp = H(k, i);
                H(k, i) = c(k) * temp + s(k) * H(k+1, i);
                H(k+1, i) = -s(k) * temp + c(k) * H(k+1, i);
            end
            [c(i), s(i)] = givens(H(i, i), H(i+1, i));
            % Apply Givens rotations to update residual norm
            H(i, i) = c(i) * H(i, i) + s(i) * H(i+1, i);
            beta = -s(i) * H(i+1, i);
            H(i+1, i) = 0;
            resvec(end+1) = abs(beta);
            if abs(beta) < tol
                flag = 0;
                iter = iter + 1;
                relres = norm(r) / norm(b);
                return;
            end
        end
        % Solve the least squares problem
        y = H(1:block_size, 1:block_size) \ (beta * eye(block_size, 1));
        x = x + V(:, 1:block_size) * y;
        r = b - A*x;
        beta = norm(r);
        iter = iter + 1;
        resvec(end+1) = beta;
    end
    relres = norm(r) / norm(b);
end

function [c, s] = givens(a, b)
    if b == 0
        c = 1;
        s = 0;
    else
        if abs(b) > abs(a)
            tau = -a / b;
            s = 1 / sqrt(1 + tau^2);
            c = s * tau;
        else
            tau = -b / a;
            c = 1 / sqrt(1 + tau^2);
            s = c * tau;
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

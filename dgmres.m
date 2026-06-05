function [x,resvec] = dgmres(A,b,m,tol,maxit,P,x0,p)
%Flexible GMRES (FGMRES) with restart length m.
%Solves Ax = b to tolerance tol using preconditioner P and initial guess x0

%Reference:
%Saad, "A flexible inner-outer preconditioned GMRES algorithm", SIAM 1993

N = size(b,1);
Ip = eye(N);
%fIp = rot90(Ip, 1);
resvec = zeros(maxit);

nit = 1;

while (nit <= maxit)
    H = zeros(m+1,m);
    r0 = b - A*x0;
    beta = norm(r0);
    V = zeros(N, m+1);
    V(:,1) = (1/beta)*r0;
    Z = zeros(N,m);
    for j=1:m
        if (j <= m - p)
            Z(:,j) = P*V(:,j);
        end
        if (j > m - p)
            Z(:,j) = P*Ip(:,m-j+1);
        end
        w = A*Z(:,j);
        for i=1:j
            H(i,j) = w'*V(:,i);
            w = w - H(i,j)*V(:,i);
        end
        H(j+1, j) = norm(w);
        V(:, j+1) = (1/H(j+1, j))*w;
    
    
        e1 = zeros(j+1,1); e1(1) = 1;
        y = H(1 : j+1, 1:j)\(beta*e1);
        x = x0 + Z(:, 1:j)*y;
        
        resnorm = norm(b-A*x);
        resvec(nit) = resnorm;

        if (resnorm < tol*norm(b))
            resvec = resvec(1:nit);
            disp(['DGMRES converged to relative tolerance ', ...
            num2str(resnorm/norm(b)), ' at iteration ', num2str(nit)])
            return
        end

        nit = nit + 1;
        if (nit > maxit)
            break;
        end
    end
    
    x0 = x;
end
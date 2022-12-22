function [time, rvec1, rvec2] = gmresTest(n, useIL0, useSam=false)


    A = sparse(diag(2*ones(1,n))) + sparse(diag(-1*ones(1,n-1),1)) + sparse(diag(-1*ones(1,n-1),-1));
    A2 = sparse(diag(3*ones(1,n))) + sparse(diag(-1*ones(1,n-1),1)) + sparse(diag(-1*ones(1,n-1),-1));

    b = zeros(n, 1);
    b(1) = 1.0;
    
    tic;
    if useIL0
        [L, U] = ilu(A);
        [~, ~, ~, ~, rvec1] = gmres(A, b, n, 1e-8, n, L, U, zeros(n, 1));
        if useSam
            N = sam(A,A2, A);
            U = U * N; 
        else
            [L, U] = ilu(A2);
        end
        [~, ~, ~, ~, rvec2] = gmres(A2, b, n, 1e-8, n, L, U, zeros(n, 1));
    else
        [~, ~, ~, ~, rvec1] = gmres(A, b, n, 1e-8, n, [], [], zeros(n, 1));
        [~, ~, ~, ~, rvec2] = gmres(A2, b, n, 1e-8, n, [], [], zeros(n, 1));
    end
    time = toc;

end

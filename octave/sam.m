function N = sam(S, Ak, A0)
    [rows, n] = size(A0);
    maxSk = 0;
    maxRk = 0;
    r = cell(n);
    s = cell(n);
    for k = 1:n
        s{k} = find(S(:,k));
        nnz(k) = length(s{k});
        for i = 1:nnz(k)
            j = s{k}(i);
            r{k} = union(r{k}, find(Ak(:,j)));
        end
        maxSk = max(nnz(k), maxSk);
        maxRk = max(length(r{k}), maxSk);
    end


    cnt = 0;
    for k = 1:n
        Atmp = Ak(r{k}, s{k});
        f = A0(r{k}, k);
        [Q, R] = qr(Atmp);
        z = R \ (Q' * f);
        idx = (cnt + 1):(cnt + nnz(k));
        rowN(idx) = s{k};
        colN(idx) = k;
        valN(idx) = z;
        cnt = cnt + nnz(k);
    end
    
    N = sparse(rowN, colN, valN);
end

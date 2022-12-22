#pragma once

template<MemLoc M, MemLoc M2>
std::pair<Vector<M>, Vector<M>> arnoldi(SparseMatrix<M2>& A, ColMajMatrix<M>& Q, int k) { // for col 0, k == 0
    Vector<M> Qcolk = Q.col(k);
    Vector<M> q = A * Qcolk;

    Vector<M> h{k + 2};

    for (int i = 0; i <= k; i++) {
        Vector<M> Qcoli = Q.col(i);
        h[i] = q.dot(Qcoli);
        q.scaledAdd(-h[i], Qcoli);
    }

    float norm = q.norm();
    assert(h.rows() > k + 1);
    h[k+1] = norm; 
    
    q /= norm;

    return {q, h};
}

template<>
std::pair<Vector<DEVICE>, Vector<DEVICE>> arnoldi<DEVICE>(SparseMatrix<DEVICE>& A, ColMajMatrix<DEVICE>& Q, int k) { // for col 0, k == 0
    Vector<DEVICE> Qcolk = Q.col(k);
    Vector<DEVICE> q = A * Qcolk;

    Vector<HOST> h{k + 2};

    for (int i = 0; i <= k; i++) {
        Vector<DEVICE> Qcoli = Q.col(i);
        h[i] = q.dot(Qcoli);
        q.addScaled(-h[i], Qcoli);
    }

    float norm = q.norm();
    assert(h.rows() > k + 1);
    h[k+1] = norm; 
    
    q /= norm;

    Vector<DEVICE> hdev = h.copy<DEVICE>();

    return {std::move(q), std::move(hdev)};
}

void specialArnoldi(SparseMatrix<DEVICE>& A, ColMajMatrix<DEVICE>& Q, ColMajMatrix<HOST>&H, int k) { // for col 0, k == 0
    Vector<DEVICE> Qcolk = Q.col(k);
    Vector<DEVICE> Qkp1 = A * Qcolk;
    Q.setCol<DEVICE>(k + 1, Qkp1);

    Vector<DEVICE> q = Q.col(k + 1);
   
    // TODO opportunity for fusion and parallelism here 
    for (int i = 0; i <= k; i++) {
        Vector<DEVICE> Qcoli = Q.col(i);
        H(i, k) = q.dot(Qcoli);
        q.addScaled(-H(i, k), Qcoli);
    }

    float norm = q.norm();
    H(k+1, k) = norm; 
    q /= norm;
}

void precondArnoldi(SparseMatrix<DEVICE>& A, ColMajMatrix<DEVICE>& Q, ColMajMatrix<HOST>&H,int k, const SparseILU<DEVICE>& M) { 
    Vector<DEVICE> Qcolk = Q.col(k);
    Vector<DEVICE> q = M.applyInv(A * Qcolk);
    Q.setCol<DEVICE>(k + 1, q);

    q = Q.col(k + 1);

    for (int i = 0; i <= k; i++) {
        Vector<DEVICE> Qcoli = Q.col(i);
        H(i, k) = q.dot(Qcoli);
        q.addScaled(-H(i, k), Qcoli);
    }

    float norm = q.norm();
    H(k+1, k) = norm; 
    
    q /= norm;
}

void precondSamArnoldi(SparseMatrix<DEVICE>& A, ColMajMatrix<DEVICE>& Q, ColMajMatrix<HOST>&H,int k, const SparseILU<DEVICE>& M, const SparseMatrix<DEVICE>& Nk) { 
    Vector<DEVICE> Qcolk = Q.col(k);
    Vector<DEVICE> q = M.applyInv(A * Qcolk);
    q = Nk * q;
    Q.setCol<DEVICE>(k + 1, q);

    q = Q.col(k + 1);

    for (int i = 0; i <= k; i++) {
        Vector<DEVICE> Qcoli = Q.col(i);
        H(i, k) = q.dot(Qcoli);
        q.addScaled(-H(i, k), Qcoli);
    }

    float norm = q.norm();
    H(k+1, k) = norm; 
    
    q /= norm;
}


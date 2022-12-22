#pragma once

std::pair<float, float> givens_rotation(float a, float b) {
    float t = std::sqrt(a * a + b * b);
    float cs = a / t;
    float sn = b / t;
    return {cs, sn};
}

template<MemLoc M>
std::pair<float, float> apply_givens(ColMajMatrix<HOST>& H, int colH, Vector<M>& cs, Vector<M>& sn, int k) {

    float cs_k, sn_k;

    for(int i = 0; i < k - 1; i++) {
        float H0 = H(i, colH);
        float H1 = H(i+1, colH);
        float cs_i = cs[i];
        float sn_i = sn[i];
        H(i, colH) = cs_i * H0 + sn_i * H1;
        H(i+1,colH) = -sn_i * H0 + cs_i * H1;
    }

    auto p = givens_rotation(H(k, colH), H(k + 1, colH));

    cs_k = p.first;
    sn_k = p.second;

    H(k, colH) = cs_k * H(k, colH) + sn_k * H(k + 1, colH);
    H(k + 1, colH) = 0;

    return {cs_k, sn_k};
}

template<>
std::pair<float, float> apply_givens<HOST>(ColMajMatrix<HOST>& H, int colH, Vector<HOST>& cs, Vector<HOST>& sn, int k) {

    float cs_k, sn_k;

    for(int i = 0; i < k - 1; i++) {
        float H0 = H(i, colH);
        float H1 = H(i+1, colH);
        float cs_i = cs[i];
        float sn_i = sn[i];
        H(i, colH) = cs_i * H0 + sn_i * H1;
        H(i+1,colH) = -sn_i * H0 + cs_i * H1;
    }

    auto p = givens_rotation(H(k, colH), H(k + 1, colH));

    cs_k = p.first;
    sn_k = p.second;

    H(k, colH) = cs_k * H(k, colH) + sn_k * H(k + 1, colH);
    H(k + 1, colH) = 0;

    return {cs_k, sn_k};
}



#include "operator.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

static real_t DROPOUT_THRESHOLD = 0.99;
void reset(real_t *input, int_t size)
{
    for (int_t i = 0; i < size; ++i) { input[i] = 0; }
}

void non_linear_forward_operator(real_t *bottom, int_t size, real_t *top)
{
    for (int_t i = 0; i < size; ++i) { 
        top[i] = non_linear_function(bottom[i]);
    }
}

void non_linear_backward_operator(real_t *diff_top, int_t size, real_t * bottom,
    real_t * diff_bottom)
{
    for (int_t i = 0; i < size; ++i) {
        diff_bottom[i] = diff_top[i] * diff_non_linear_function(bottom[i]);
    }
}

void non_linear_forward(real_t **bottoms, int_t nb, int_t size, real_t **tops)
{
    for (int_t i = 0; i < nb; ++i) {
        non_linear_forward_operator(bottoms[i], size, tops[i]);
    }
}

void non_linear_backward(real_t **diff_tops, int_t nt, int_t size,
    real_t ** bottoms, real_t ** diff_bottoms)
{
    for (int_t i = 0; i < nt; ++i) {
        non_linear_backward_operator(diff_tops[i], size, bottoms[i],
                                     diff_bottoms[i]);
    }
}

void convolution_forward_operator(real_t * bottom_M, int_t nbr, int_t nbc,
    real_t * weight_M, int_t nwr, int_t wsz, real_t * top_M)
{
    int_t i1, i2, i3, i4, i5, wlb;
    const int_t ntc = wsz + nbc - 1;      //number of result matrix columns
    const int_t nwc = wsz * nbr + 1;      //number of weight matrix columns
    real_t z = 0;                         //weight sum.
    for (i1 = 0; i1 < ntc; ++i1){
        for (i2 = 0; i2 < nwr; ++i2){
            i3 = i1;
            i4 = nwc - 2;
            if (i3 >= nbc){
                i4 -= (i3 - nbc + 1) * nbr;
                i3 = nbc - 1;
            }
            for (z = 0, wlb = i1 - wsz; i3 > -1 && i3 > wlb; --i3) {
                for (i5 = nbr - 1; i5 > -1; --i5, --i4) {
                    z += weight_M[i2 * nwc + i4] * bottom_M[i5 * nbc + i3];
                }
            }
            top_M[i2 * ntc + i1] += z + weight_M[(1 + i2) * nwc - 1];// bias
        }
    }
}

void convolution_backward_operator(real_t * diff_top_M, int_t ntr, int_t ntc,
    real_t * weight_M, real_t * diff_weight_M, int_t wsz, real_t * bottom_M,
    real_t * diff_bottom_M, int_t nbr)
{
    const int_t nwc = wsz * nbr + 1;    //column number of weight matrix.
    const int_t tmp = nwc * ntr;        //size of weight matrix.
    const int_t nbc = ntc + 1 - wsz;    //number of result matrix columns
    int_t i1, i2, i3, i4, i5, wlb;
    real_t z = 0;
    for (i1 = 0; i1 < ntc; ++i1){
        i2 = i1;
        i3 = nwc - 2;
        if (i2 >= nbc){
            i3 -= (i2 - nbc + 1) * nbr;
            i2 = nbc - 1;
        }
        for (wlb = i1 - wsz; i2 > -1 && i2 > wlb; --i2) {
            for (i4 = nbr - 1; i4 > -1; --i4, --i3){
                for (z = 0, i5 = 0; i5 < ntr; ++i5){
                    diff_weight_M[i5 * nwc + i3] += diff_top_M[i5 * ntc + i1]
                        * bottom_M[i4 * nbc + i2]; //weight
                    z += diff_top_M[i5 * ntc + i1] * weight_M[i5 * nwc + i3];
                }
                diff_bottom_M[i4 * nbc + i2] += z;
            }
        }
        for (i5 = 0; i5 < ntr; ++i5) { // loop for bias
            diff_weight_M[(i5 + 1) * nwc - 1] += diff_top_M[i5 * ntc + i1];
        }
    }
}

//weight_M nbm * ntm
void convolution_forward(real_t ** bottom_Ms, int_t nbm, int_t nbr, int_t nbc,
    real_t **weight_Ms, int_t nwr, int_t wsz, real_t ** top_Ms, int_t ntm)
{
    const int_t ntc = wsz + nbc - 1;
    int_t i1, i2;
    for (i2 = 0; i2 < ntm; ++i2) { reset(top_Ms[i2], nwr * ntc); }
    for (i1 = 0; i1 < nbm; ++i1) {
        for (i2 = 0; i2 < ntm; ++i2) {
            convolution_forward_operator(bottom_Ms[i1], nbr, nbc,
                weight_Ms[i1 * ntm + i2], nwr, wsz, top_Ms[i2]);
        }
    }
}

//weight_M nbm * ntm
void convolution_backward(real_t ** diff_top_Ms, int_t ntm, int_t ntr,
    int_t ntc, real_t ** weight_Ms, real_t ** diff_weight_Ms, int_t wsz,
    real_t ** bottom_Ms, real_t ** diff_bottom_Ms, int_t nbm, int_t nbr)
{
    const int_t nbc = ntc + 1 - wsz;    //number of result matrix columns
    int_t i1, i2;
    for (i1 = 0; i1 < nbm; ++i1) { reset(diff_bottom_Ms[i1], nbr * nbc); }
    for (i1 = 0; i1 < nbm; ++i1) {
        for (i2 = 0; i2 < ntm; ++i2) {
            convolution_backward_operator(diff_top_Ms[i2], ntr, ntc,
                weight_Ms[i1 * ntm + i2], diff_weight_Ms[i1 * ntm + i2], wsz,
                bottom_Ms[i1], diff_bottom_Ms[i1], nbr);
        }
    }
}

void full_connection_forward(real_t * bottom_V, int_t nbr, real_t * weight_M,
    int_t nwr, real_t * top_V)
{
    const int_t nwc = nbr + 1;        //number of weight matrix columns
    int_t i1, i2;
    real_t z = 0;
    for (i1 = 0; i1 < nwr; ++i1){
        for (z = 0, i2 = 0; i2 < nbr; ++i2) {
            z += weight_M[i1* nwc + i2] * bottom_V[i2];
        }
        top_V[i1] = z + weight_M[i1 * nwc + nbr];    //add bias
    }
}

void full_connection_backward(real_t * diff_top_V, int_t ntr,
    real_t * weight_M, real_t * diff_weight_M, real_t * bottom_V,
    real_t * diff_bottom_V, int_t nbr)
{
    const int_t nwc = nbr + 1;        //number of weight matrix columns
    int_t i1, i2;
    real_t z = 0;
    for (i1 = 0; i1 < nbr; ++i1) {
        for (z = 0, i2 = 0; i2 < ntr; ++i2) {
            z += weight_M[i2 * nwc + i1] * diff_top_V[i2];
            diff_weight_M[i2 * nwc + i1] += diff_top_V[i2] * bottom_V[i1];
        }
        diff_bottom_V[i1] = z;
    }
    for (i2 = 0; i2 < ntr; ++i2) {
        diff_weight_M[i2 * nwc + nbr] += diff_top_V[i2];
    }
}

//with_pooling.
void full_connection_forward(real_t ** bottom_Ms, int_t nbm, int_t nbr,
    int_t nbc, real_t * weight_M, real_t * top_V, int_t ntr)
{
    const int_t tmp = nbc * nbr;
    const int_t nwc = nbm * tmp + 1;      //number of weight matrix columns
    int_t i1, i2, i3, i4;
    real_t z = 0;
    for (i1 = 0; i1 < ntr; ++i1){
        for (z = 0, i2 = 0; i2 < nbm; ++i2) {
            for (i3 = 0; i3 < nbc; ++i3) {
                for (i4 = 0; i4 < nbr; ++i4) {
                    z +=  weight_M[i1 * nwc + i2 * tmp + i3 * nbr + i4]
                        * bottom_Ms[i2][i4 * nbc + i3];
                }
            }
        }
        top_V[i1] = z + weight_M[(i1 + 1) * nwc - 1];    //add bias
    }
}

void full_connection_backward(real_t * diff_top_V, int_t ntr, real_t * weight_M,
    real_t *diff_weight_M, real_t **bottom_Ms, real_t **diff_bottom_Ms,
    int_t nbm, int_t nbr, int_t nbc)
{
    const int_t tmp = nbc * nbr;
    const int_t nwc = nbm * tmp + 1;
    int_t i1, i2, i3, i4, i5, i6;
    real_t z = 0;
    for (i1 = 0; i1 < nbm; ++i1) {
        for (i2 = 0; i2 < nbc; ++i2) {
            for (i3 = 0; i3 < nbr; ++i3){
                i5 = i3 * nbc + i2;
                for (z = 0, i4 = 0; i4 < ntr; ++i4){
                    i6 = i4 * nwc + i1 * tmp + i2 * nbr + i3;
                    z += weight_M[i6] * diff_top_V[i4];
                    diff_weight_M[i6] += diff_top_V[i4] * bottom_Ms[i1][i5];
                }
                diff_bottom_Ms[i1][i5] = z;
            }
        }
    }
    for (i4 = 0; i4 < ntr; ++i4) {
        diff_weight_M[(i4 + 1) * nwc - 1] += diff_top_V[i4];   // update bias
    }
}

void k_max_pooling_forward_operator(real_t * bottom_M, int_t nbr, int_t nbc,
    int_t * pool_V, int_t k, real_t * top_M)
{
    int_t i1, i2;
    real_t z = 0;
    real_t dpa[MAX_SEN_LEN];      //save the lengths of max vectors in up order.
    for (i1 = 0; i1 < k; ++i1) { dpa[i1] = -1; }
    for (i1 = 0; i1 < nbc; ++i1){
        for (z = 0, i2 = 0; i2 < nbr; ++i2) {
            z += square_s(bottom_M[i2 * nbc + i1]);
        }
        for (i2 = k - 2; i2 > -1; --i2) {
            if (z > dpa[i2]) {
                dpa[i2 + 1] = dpa[i2];
                pool_V[i2 + 1] = pool_V[i2];
            }
            else break;
        }
        if (dpa[i2 + 1] < z) {
            dpa[i2 + 1] = z;
            pool_V[i2 + 1] = i1;
        }
    }
    std::sort(pool_V, pool_V + k);
    for (i1 = 0; i1 < k; ++i1) {
        for (i2 = 0; i2 < nbr; ++i2) {
            top_M[i2 * k + i1] = bottom_M[i2 * nbc + pool_V[i1]];
        }
    }
}

void k_max_pooling_backward_operator(real_t * diff_top_M, int_t * pool_V,
    int_t k, int_t nbr, int_t nbc, real_t * diff_bottom_M)
{
    reset(diff_bottom_M, nbr * nbc);
    int_t i1, i2;
    for (i1 = 0; i1 < k; ++i1) {
        for (i2 = 0; i2 < nbr; ++i2) {
            diff_bottom_M[i2 * nbc + pool_V[i1]] = diff_top_M[i2 * k + i1];
        }
    }
}

void k_max_pooling_forward(real_t ** bottom_Ms, int_t nbm, int_t nbr, int_t nbc,
    int_t ** pool_Vs, int_t k, real_t ** top_Ms)
{
    for (int_t i = 0; i < nbm; ++i) {
        k_max_pooling_forward_operator(bottom_Ms[i], nbr, nbc, pool_Vs[i], k,
                                       top_Ms[i]);
    }
}

void k_max_pooling_backward(real_t ** diff_top_Ms, int_t ntm, int_t ** pool_Vs,
    int_t k, int_t nbr, int_t nbc, real_t ** diff_bottom_Ms)
{
    for (int_t i = 0; i < ntm; ++i) {
        k_max_pooling_backward_operator(diff_top_Ms[i], pool_Vs[i], k, nbr, nbc,
                                        diff_bottom_Ms[i]);
    }
}


void bernoulli(int_t *mask, int_t size, real_t prob, ulong_t &seed)
{
    static const ulong_t RAND_GROUND = 0x00000000FFFFFFFF;
    for (int_t i = 0; i < size; ++i) {
        mask[i] = static_cast<real_t>(next_random(seed) & RAND_GROUND) / 
            static_cast<real_t>(RAND_GROUND) < prob ? 1 : 0;
 
    }
}

void drop_out_forward_operator(real_t * bottom, int_t size, int_t *mask,
    real_t *top, real_t prob, ulong_t &seed)
{
    bernoulli(mask, size, prob, seed);
    for (int_t i = 0; i < size; ++i) {
        top[i] = mask[i] ? bottom[i] : 0;
    }
}

void drop_out_backward_operator(real_t * diff_top, int_t size, int_t *mask,
    real_t *diff_bottom)
{
    for (int_t i = 0; i < size; ++i) {
        diff_bottom[i] = mask[i] ? diff_top[i] : 0;
    }
}

void drop_out_forward(real_t ** bottoms, int_t nb,  int_t size, int_t **masks,
    real_t **tops, real_t prob, ulong_t &seed, bool isTrain)
{
    if (prob > DROPOUT_THRESHOLD) { return; }
    int_t i1, i2;
    if (isTrain) {
        for (i1 = 0; i1 < nb; ++i1) {
            drop_out_forward_operator(bottoms[i1], size, masks[i1], tops[i1],
                prob, seed);
        }
    } else {
        for (i1 = 0; i1 < nb; ++i1) {
            for (i2 = 0; i2 < size; ++i2) {
                tops[i1][i2] = bottoms[i1][i2] * prob;
            }
        }
    }
}

void drop_out_backward(real_t ** diff_tops, int_t nt, int_t size, int_t **masks,
    real_t **diff_bottoms, real_t prob)
{
    if (prob > DROPOUT_THRESHOLD) { return; }
    for (int_t i = 0; i < nt; ++i) {
        drop_out_backward_operator(diff_tops[i], size, masks[i],
            diff_bottoms[i]);
    }
}


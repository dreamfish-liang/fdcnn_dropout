/*******************************************************************
*  Copyright(c) 2014
*  All rights reserved.
*
*  File Name: operator.h
*  Brief    : This file provides some basic operating
*     functions for convolution neural network.
*  Current Version: 1.0
*  Author   : Zhongping Liang
*  Date     : 2014-12-12
******************************************************************/

#ifndef OPERATOR_H
#define OPERATOR_H

#include <string>
#include <cmath>

#include "dcnn_utils.h"

inline int_t upper_s(real_t x)
{
    return x > static_cast<int_t>(x) ?
        static_cast<int_t>(x)+1 : static_cast<int_t>(x);
}

inline real_t square_s(real_t x)
{
    return x * x;
}

inline real_t abs_s(real_t x)
{
    return x < 0 ? -x : x;
}

/*  Get the larger value of x and y.
*   Parameters :
*       x           : x.
*       y           : y
*   Return :
*       int_t :The larger value of x and y.
*/
inline int_t max_s(int_t x, int_t y)
{
    return x < y ? y : x;
}

/*  Provide a non linear function for the networks: here the function is tanh.
*   Parameters :
*       x           : x.
*   Return :
*       int_t : tanh(x).
*/
inline real_t non_linear_function(real_t x)
{
    static const real_t BOUND = 20;
    if (x > BOUND) { x = BOUND; }
    if (x < -BOUND) { x = -BOUND; }
    return tanh(x);
}

inline real_t diff_non_linear_function(real_t x){

    return 1 - square_s(x);
}

void reset(real_t *input, int_t size);

void non_linear_forward(real_t **bottoms, int_t nb, int_t size, real_t **tops);
void non_linear_backward(real_t **diff_tops, int_t nt, int_t size,
    real_t ** bottoms, real_t ** diff_bottoms);

void convolution_forward(real_t ** bottom_Ms, int_t nbm, int_t nbr, int_t nbc,
    real_t **weight_Ms, int_t nwr, int_t wsz, real_t ** top_Ms, int_t ntm);
void convolution_backward(real_t ** diff_top_Ms, int_t ntm, int_t ntr,
    int_t ntc, real_t ** weight_Ms, real_t ** diff_weight_Ms, int_t wsz,
    real_t ** bottom_Ms, real_t ** diff_bottom_Ms, int_t nbm, int_t nbr);

void full_connection_forward(real_t * bottom_V, int_t nbr, real_t * weight_M,
    int_t nwr, real_t * top_V);
void full_connection_backward(real_t * diff_top_V, int_t ntr,
    real_t * weight_M, real_t * diff_weight_M, real_t * bottom_V,
    real_t * diff_bottom_V, int_t nbr);

void full_connection_forward(real_t ** bottom_Ms, int_t nbm, int_t nbr,
    int_t nbc, real_t * weight_M, real_t * top_V, int_t ntr);
void full_connection_backward(real_t * diff_top_V, int_t ntr, real_t * weight_M,
    real_t *diff_weight_M, real_t **bottom_Ms, real_t **diff_bottom_Ms,
    int_t nbm, int_t nbr, int_t nbc);

void k_max_pooling_forward(real_t ** bottom_Ms, int_t nbm, int_t nbr, int_t nbc,
    int_t ** pool_Vs, int_t k, real_t ** top_Ms);
void k_max_pooling_backward(real_t ** diff_top_Ms, int_t ntm, int_t ** pool_Vs,
    int_t k, int_t nbr, int_t nbc, real_t ** diff_bottom_Ms);

void drop_out_forward(real_t ** bottoms, int_t nb, int_t size, int_t **masks,
    real_t **tops, real_t prob, ulong_t &seed, bool isTrain);
void drop_out_backward(real_t ** diff_tops, int_t nt, int_t size, int_t **masks,
    real_t **diff_bottoms, real_t prob);

inline ulong_t next_random(ulong_t &seed)
{
    seed = (seed * 25214903917UL + 11UL);
    return seed;
}

#endif 

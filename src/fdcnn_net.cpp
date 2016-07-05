/*==============================================================================
 *   Copyright (C) 2016 All rights reserved.
 *
 *  File Name   : fdcnn_net.h
 *  Author      : Zhongping Liang
 *  Date        : 2016-07-05
 *  Version     : 1.0
 *  Description : This file provides implements for FdcnnNet.
 *============================================================================*/

#include "fdcnn_net.h"

#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <functional>
#include <cfloat>
#include <unistd.h>

#include "string_handler.h"
#include "instance.h"
namespace fdcnn
{

using std::pair;
using std::vector;
using std::string;
using std::ifstream;
using std::fstream;
using std::istringstream;
using std::ostringstream;

FdcnnNet::FdcnnNet()
{
    nsr = 0;
    ncl = 0;
    scw = 0;
    ncr = 0;
    MM = 0;
    ncm = 0;
    Ktop = 0;
    nhl = 0;
    nhr = 0;
    WM = 0;
    U = 0;
    nor = 0;
    nfd = 0;
    nif = 0;
    feature_table = 0;
    diff_MM = 0;
    diff_WM = 0;
    diff_U = 0;
    sen_M = 0;
    diff_sen_M = 0;
    lsen = 0;
    ncc = 0;
    ndk = 0;
    conv_M = 0;
    diff_conv_M = 0;
    dropout_conv_mask_M = 0;
    pool_V = 0;
    pool_M = 0;
    diff_pool_M = 0;
    hid_V = 0;
    diff_hid_V = 0;
    dropout_hid_mask_V = 0;
    out_V = 0;
    diff_out_V = 0;
    dropout_prob = 0.5;
    thrd_capacity = 0;
    batch_size = 0;
    snapshot = 500;
    alpha = 0.01;
    lambda = 0;
    isUsingWord2vec = 0;
    iter_num = 10000;
    bin = 1;
    busy_num = 0;
    avgerr = 0;
    cur_iter = 0;

    pthread_mutex_init(&busy_num_mutex, NULL);
    pthread_mutex_init(&cur_iter_mutex, NULL);
}

FdcnnNet::~FdcnnNet()
{
    freeRunMemory();
    freeWeightMemory();
    freeFeatureMemory();
    freeConfigMemory();
}

const int_t  FdcnnNet::LOG_PER_ITERS   = 100;    //log per 100 iters
const uint_t FdcnnNet::USLEEP_INTERVAL = 1000;   //1 ms

void FdcnnNet::LookupTable(InstancePtr instance, real_t *sen_M)
{
    LOG_DEBUG(("Enter LookupTable"));
    int_t i, j, k, l;
    int_t lsen= instance->GetSenLen();
    StrToIntMap::iterator it;
    for (i = 0; i < lsen; ++i) {
        for (j = 0, k = 0; j < nif; k += nfd[j], ++j) {
            it = feat_dicts[j].find(instance->features[j][i]);
            if (feat_dicts[j].end() == it) {
                it = feat_dicts[j].find(NIL);
            }
            for (l = 0; l < nfd[j]; ++l) {
                sen_M[(l + k) * lsen + i] 
                    = feature_table[j][it->second * nfd[j] + l];
            }
        }
    }
    LOG_DEBUG(("Leave LookupTable"));
}

bool FdcnnNet::ReadConfig(const char * filename){
    FILE * config_file = fopen(filename, "rb");
    if (!config_file) {
        LOG_ERROR(("Open config file failed, filename: ")(filename));
        return false;
    }
    ScopedFile scoped_file(config_file);
    LOG_INFO(("Start reading config"));
    freeConfigMemory();
    int_t i;
    fscanf(config_file, "dropout = %lf\n", &dropout_prob);
    fscanf(config_file, "nif = %d\n", &nif);
    nfd = new int_t[nif];
    fscanf(config_file, "nfd = ");
    nsr = 0;
    for (i = 0; i < nif; ++i) {
        fscanf(config_file, "%d ", &nfd[i]);
        nsr += nfd[i];
    }
    fscanf(config_file, "ncl = %d\n", &ncl);
    scw = new int_t[ncl];
    ncr = new int_t[ncl];
    ncm = new int_t[ncl];
    fscanf(config_file, "scw = ");
    for (i = 0; i < ncl; ++i) { fscanf(config_file, "%d ", &scw[i]); }
    fscanf(config_file, "\nncr = ");
    for (i = 0; i < ncl; ++i) { fscanf(config_file, "%d ", &ncr[i]); }
    fscanf(config_file, "\nncm = ");
    for (i = 0; i < ncl; ++i) { fscanf(config_file, "%d ", &ncm[i]); }
    fscanf(config_file, "\nKtop = %d", &Ktop);
    fscanf(config_file, "\nnhl = %d", &nhl);
    nhr = new int_t [nhl];
    fscanf(config_file, "\nnhr = ");
    for (i = 0; i < nhl; ++i) { fscanf(config_file, "%d ", &nhr[i]); }
    fscanf(config_file, "\n");
    logInfoConfig();
    LOG_INFO(("Finish reading config"));
    return true;
}

bool FdcnnNet::CheckConfig(){
    int_t i, flag = 0;
    if (dropout_prob < 0 || dropout_prob > 1) {
        LOG_ERROR(("Wrong dropout value: ")(dropout_prob));
        flag = 1;
    }
    if (nif < 1){
        LOG_ERROR(("Number of features can not be small than 1."));
        flag = 1;
    }
    if (nif > MAX_FIELD_NUM){
        LOG_ERROR(("Number of features can not exceed: ")(MAX_FIELD_NUM));
        flag = 1;
    }
    for (i = 0; i < nif; ++i) if (nfd[i] < 1) {
        LOG_ERROR(("Dimension of the ")(i + 1)
            ("th features can not be small than 1."));
        flag = 1;
    }
    if (ncl < 1){
        LOG_ERROR(("Number of convolution layers can not be small than 1."));
        flag = 1;
    }
    for (i = 0; i < ncl; ++i) if (scw[i] < 1){
        LOG_ERROR(("Size of window in the ")(i + 1)
            ("th layer can not be small than 1."));
        flag = 1;
    }
    for (i = 0; i < ncl; ++i) if (ncr[i] < 1) {
        LOG_ERROR(("Dimension of rows in the ")(i + 1)("th convolution"
                " layer can not be small than 1."));
        flag = 1;
    }
    for (i = 0; i < ncl; ++i) if (ncm[i] < 1){
        LOG_ERROR(("Number of convolutions in the ")(i + 1)("th convolution"
            " layer can not be small than 1."));
        flag = 1;
    }
    if (Ktop < 1) {
        LOG_ERROR(("Ktop can not be small than 1."));
        flag = 1;
    }
    if (nhl < 1) {
        LOG_ERROR(("Number of hidden layers can not be small than 1."));
        flag = 1;
    }
    for (i = 0; i < nhl; ++i) if (nhr[i] < 1) {
        LOG_ERROR(("Dimension of rows in the ")(i + 1)("th hidden layer "
            "can not be small than 1."));
        flag = 1;
    }
    if (flag) { return false; }
    return true;
}

void FdcnnNet::allocWeightMemory() {
    LOG_DEBUG(("Enter AllocParamMemory"));
    LOG_INFO(("Start alloc param memory"));
    freeWeightMemory();
    int_t i, j, tmp;
    MM = new real_t**[ncl];
    MM[0] = new real_t*[ncm[0]];
    for (j = 0; j < ncm[0]; ++j) {
        MM[0][j] = new real_t[(ncr[0] * (scw[0] * nsr + 1))];
    }
    for (i = 1; i < ncl; ++i) {
        tmp = ncm[i - 1] * ncm[i];
        MM[i] = new real_t*[tmp];
        for (j = 0; j < tmp; ++j) {
            MM[i][j] = new real_t[ncr[i] * (scw[i] * ncr[i - 1] + 1)];
        }
    }
    WM = new real_t*[nhl];
    WM[0] = new real_t[nhr[0] * (Ktop * ncr[ncl - 1] * ncm[ncl - 1] + 1)];
    for (i = 1; i < nhl; ++i) {
        WM[i] = new real_t[nhr[i] * (nhr[i - 1] + 1)];
    }
    U = new real_t[nor * (nhr[nhl - 1] + 1)];
    LOG_INFO(("Finish alloc param memory"));
    LOG_DEBUG(("Leave AllocParamMemory"));
}

void FdcnnNet::allocFeatureMemory(){
    LOG_DEBUG(("Enter allocFeatureMemory"));
    LOG_INFO(("Start alloc feature memory"));
    freeFeatureMemory();
    feature_table = new real_t*[nif];
    for (int_t i = 0; i < nif; ++i) {
        feature_table[i] = new real_t[feat_dicts[i].size() *  nfd[i]];
    }
    LOG_INFO(("Finish alloc feature memory"));
    LOG_DEBUG(("Leave allocFeatureMemory"));
}

void FdcnnNet::freeFeatureMemory()
{
    LOG_DEBUG(("Enter freeFeatureMemory"));
    LOG_INFO(("Start free feature memory"));
    if (feature_table) {
        for (int_t i = 0; i < nif; ++i) { delete[] feature_table[i]; }
        delete[] feature_table;
        feature_table = 0;
    }
    LOG_INFO(("Finish free feature memory"));
    LOG_DEBUG(("Leave freeFeatureMemory"));
}

void FdcnnNet::freeWeightMemory(){
    LOG_DEBUG(("Enter FreeParamMemory"));
    int_t i, j, tmp;
    if (MM) {
        for (j = 0; j < ncm[0]; ++j) { delete[] MM[0][j]; }
        delete[] MM[0];
        for (i = 1; i < ncl; ++i) {
            tmp = ncm[i - 1] * ncm[i];
            for (j = 0; j < tmp; ++j) { delete[] MM[i][j]; }
            delete[] MM[i];
        }
        delete[] MM;
        MM = 0;
    }
    if (WM) {
        for (i = 0; i < nhl; ++i) { delete[] WM[i]; }
        delete[] WM;
        WM = 0;
    }
    if (U) {
        delete[] U;
        U = 0;
    }
    LOG_DEBUG(("Leave FreeParamMemory"));
}

void FdcnnNet::freeConfigMemory(){
    LOG_DEBUG(("Enter FreeParamMemory"));
    if (nfd) {
        delete[] nfd;
        nfd = 0;
    }
    if (scw) {
        delete[] scw;
        scw = 0;
    }
    if (ncr) {
        delete[] ncr;
        ncr = 0;
    }
    if (ncm) {
        delete[] ncm;
        ncm = 0;
    }
    if (nhr) {
        delete[] nhr;
        nhr = 0;
    }
    LOG_DEBUG(("Leave FreeParamMemory"));
}

void FdcnnNet::Setup() {
    LOG_DEBUG(("Enter Setup"));
    LOG_INFO(("Start Setuping"));

    int_t i, j, k, tmp1, tmp2, layer_size, size;
    layer_size = scw[0] * nsr + 1;
    tmp2 = ncr[0] * layer_size;
    for (j = 0; j < ncm[0]; ++j) {
        for (k = 0; k < tmp2; ++k) { MM[0][j][k] = getRand(layer_size); }
    }
    for (i = 1; i < ncl; ++i) {
        tmp1 = ncm[i - 1] * ncm[i];
        layer_size = scw[i] * ncr[i - 1] + 1;
        tmp2 = ncr[i] * layer_size;
        for (j = 0; j < tmp1; ++j) {
            for (k = 0; k < tmp2; ++k) { MM[i][j][k] = getRand(layer_size); }
        }
    }
    layer_size = Ktop * ncr[ncl - 1] * ncm[ncl - 1] + 1;
    tmp1 = nhr[0] * layer_size;
    for (j = 0; j < tmp1; ++j) { WM[0][j] = getRand(layer_size); }
    for (i = 1; i < nhl; ++i) {
        layer_size = nhr[i - 1] + 1;
        tmp1 = nhr[i] * layer_size;
        for (j = 0; j < tmp1; ++j) { WM[i][j] = getRand(layer_size); }
    }
    layer_size = nhr[nhl - 1] + 1;
    tmp1 = nor * layer_size;
    for (j = 0; j < tmp1; ++j) { U[j] = getRand(layer_size); }
    for (i = 0; i < nif; ++i) {
        for (k = 0; k < nfd[i]; ++k) { feature_table[i][k] = 0; }
        size = static_cast<int_t>(feat_dicts[i].size());
        for (j = 1; j < size; ++j) {
            for (k = 0; k < nfd[i]; ++k) {
                feature_table[i][j * nfd[i] + k] = getRand(nfd[i]);
            }
        }
    }
    LOG_INFO(("Finish Setuping"));
    LOG_DEBUG(("Leave Setup"));
}

void FdcnnNet::Forward(real_t * sen_M, int_t lsen, real_t***conv_M, int_t * ncc,
    int_t ***dropout_conv_mask_M, int_t ***pool_V, int_t *ndk, real_t***pool_M,
    real_t ** hid_V, int_t **dropout_hid_mask_V, real_t *out_V, ulong_t &seed,
    bool isTrain)
{
    LOG_DEBUG(("Enter Forward"));
    //sentence layer to convolution and pooling layers. 
    ncc[0] = lsen + scw[0] - 1;
    convolution_forward(&sen_M, 1, nsr, lsen, MM[0], ncr[0], scw[0], conv_M[0],
                        ncm[0]);
    non_linear_forward(conv_M[0], ncm[0], ncr[0] * ncc[0], conv_M[0]);
    if (isTrain) {
        dropout_forward(conv_M[0], ncm[0], ncr[0] * ncc[0],
            dropout_conv_mask_M[0], conv_M[0], dropout_prob, seed, true);
    } else {
        dropout_forward(conv_M[0], ncm[0], ncr[0] * ncc[0], 0, conv_M[0],
            dropout_prob, seed, false);
    }
    k_max_pooling_forward(conv_M[0], ncm[0], ncr[0], ncc[0], pool_V[0],
        ndk[0], pool_M[0]);

    int_t i, tmp;
    // convolution layers
    for (i = 1, tmp = i - 1; i < ncl; ++i, ++tmp){
        ncc[i] = ndk[tmp] + scw[i] - 1;
        convolution_forward(pool_M[tmp], ncm[tmp], ncr[tmp], ndk[tmp], MM[i],
            ncr[i], scw[i], conv_M[i], ncm[i]);
        non_linear_forward(conv_M[i], ncm[i], ncr[i] * ncc[i], conv_M[i]);
        if (isTrain) {
            dropout_forward(conv_M[i], ncm[i], ncr[i] * ncc[i],
                dropout_conv_mask_M[i], conv_M[i], dropout_prob, seed, true);
        } else {
            dropout_forward(conv_M[i], ncm[i], ncr[i] * ncc[i], 0, conv_M[i],
                dropout_prob, seed, false);
        }
        k_max_pooling_forward(conv_M[i], ncm[i], ncr[i], ncc[i], pool_V[i],
            ndk[i], pool_M[i]);
    }
    //pooling to 0 hidden layer
    tmp = ncl - 1;
    full_connection_forward(pool_M[tmp], ncm[tmp], ncr[tmp], ndk[tmp], WM[0],
        hid_V[0], nhr[0]);
    non_linear_forward(&(hid_V[0]), 1, nhr[0], &(hid_V[0]));
    if (isTrain) {
        dropout_forward(&(hid_V[0]), 1, nhr[0], &(dropout_hid_mask_V[0]),
            &(hid_V[0]), dropout_prob, seed, true);
    } else {
        dropout_forward(&(hid_V[0]), 1, nhr[0], 0, &(hid_V[0]), dropout_prob,
            seed, false);
    }

    //hidden 0 to hidden top
    for (i = 1, tmp = i - 1; i < nhl; ++i, ++tmp) {
        full_connection_forward(hid_V[tmp], nhr[tmp], WM[i], nhr[i],
            hid_V[i]);
        non_linear_forward(&(hid_V[i]), 1, nhr[i], &(hid_V[i]));
        if (isTrain) {
            dropout_forward(&(hid_V[i]), 1, nhr[i], &(dropout_hid_mask_V[i]),
                &(hid_V[i]), dropout_prob, seed, true);
        } else {
            dropout_forward(&(hid_V[i]), 1, nhr[i], 0, &(hid_V[i]),
                dropout_prob, seed, false);
        }
    }
    //hidden top to out layer
    full_connection_forward(hid_V[tmp], nhr[tmp], U, nor, out_V);
    LOG_DEBUG(("Leave Forward"));
}

void FdcnnNet::Backward(real_t * sen_M, real_t *diff_sen_M, int_t lsen, 
    real_t ***conv_M, real_t ***diff_conv_M, int_t *ncc, 
    int_t ***dropout_conv_mask_M, int_t ***pool_V, int_t *ndk, 
    real_t*** pool_M, real_t*** diff_pool_M, real_t **hid_V,
    real_t**diff_hid_V, int_t **dropout_hid_mask_V, real_t * diff_out_V,
    real_t ***diff_MM, real_t **diff_WM, real_t *diff_U)
{
    LOG_DEBUG(("Enter Backward"));
    const int_t htop = nhl - 1;
    const int_t ctop = ncl - 1;
    int_t i, tmp;
    //out to hidden top
    full_connection_backward(diff_out_V, nor, U, diff_U, hid_V[htop],
        diff_hid_V[htop], nhr[htop]);
    //hidden top to hidden 0
    for (i = htop, tmp = i - 1; i > 0; --i, --tmp) {
        dropout_backward(&(diff_hid_V[i]), 1, nhr[i],
            &(dropout_hid_mask_V[i]), &(diff_hid_V[i]), dropout_prob);
        non_linear_backward(&(diff_hid_V[i]), 1, nhr[i], &(hid_V[i]),
            &(diff_hid_V[i]));
        full_connection_backward(diff_hid_V[i], nhr[i], WM[i], diff_WM[i],
            hid_V[tmp], diff_hid_V[tmp], nhr[tmp]);
    }
    //hidden 0 to cm top
    dropout_backward(&(diff_hid_V[0]), 1, nhr[0],
        &(dropout_hid_mask_V[0]), &(diff_hid_V[0]), dropout_prob);
    non_linear_backward(&(diff_hid_V[0]), 1, nhr[0], &(hid_V[0]),
        &(diff_hid_V[0]));

    full_connection_backward(diff_hid_V[0], nhr[0], WM[0], diff_WM[0],
        pool_M[ctop], diff_pool_M[ctop], ncm[ctop], ncr[ctop], ndk[ctop]);

    //cm top to cm 0
    for (i = ctop, tmp = i - 1; i > 0; --i, --tmp) {
        k_max_pooling_backward(diff_pool_M[i], ncm[i], pool_V[i], ndk[i],
            ncr[i], ncc[i], diff_conv_M[i]);
        dropout_backward(diff_conv_M[i], ncm[i], ncr[i] * ncc[i],
            dropout_conv_mask_M[i], diff_conv_M[i], dropout_prob);
        non_linear_backward(diff_conv_M[i], ncm[i], ncr[i] * ncc[i],
            conv_M[i], diff_conv_M[i]);
        convolution_backward(diff_conv_M[i], ncm[i], ncr[i], ncc[i], MM[i],
            diff_MM[i], scw[i], pool_M[tmp], diff_pool_M[tmp], ncm[tmp],
            ncr[tmp]);
    }
    // cm 0 to sen
    k_max_pooling_backward(diff_pool_M[0], ncm[0], pool_V[0], ndk[0],
        ncr[0], ncc[0], diff_conv_M[0]);
    dropout_backward(diff_conv_M[0], ncm[0], ncr[0] * ncc[0],
        dropout_conv_mask_M[0], diff_conv_M[0], dropout_prob);
    non_linear_backward(diff_conv_M[0], ncm[0], ncr[0] * ncc[0],
        conv_M[0], diff_conv_M[0]);
    convolution_backward(diff_conv_M[0], ncm[0], ncr[0], ncc[0], MM[0],
        diff_MM[0], scw[0], &sen_M, &diff_sen_M, 1, nsr);
    LOG_DEBUG(("Leave Backward"));
}

void FdcnnNet::MakeDict(const std::vector<InstancePtr> &instances)
{
    LOG_DEBUG(("Enter MakeDict"));
    LOG_INFO(("Start making dict."));
    vector<int_t> feat_ids(nif, 0);
    int_t label_id = 0;
    int_t i, j, lsen;
    feat_dicts.resize(nif);
    for (i = 0; i < nif; ++i) {
        feat_dicts[i].insert(StrToIntMap::value_type(NIL, feat_ids[i]++));
    }
    for (std::vector<InstancePtr>::const_iterator cit = instances.begin();
        instances.end() != cit; ++cit) {
        lsen = (*cit)->GetSenLen();
        for (i = 0; i < lsen; ++i) {
            for (j = 0; j < nif; ++j) {
                if (feat_dicts[j].insert(StrToIntMap::value_type(
                    (*cit)->features[j][i], feat_ids[j])).second) {
                    ++feat_ids[j];
                }
            }
        }
        if (label_dict.insert(StrToIntMap::value_type(
            (*cit)->label, label_id)).second) {
            ++label_id;
        }
    }
    nor = static_cast<int_t>(label_dict.size());
    makeIndex2label();
    allocWeightMemory();
    allocFeatureMemory();
    LOG_INFO(("Finish making dict."));
    LOG_DEBUG(("Leave MakeDict"));
}

bool FdcnnNet::LoadWord2vec(const char * filename) 
{
    ifstream ifs(filename, fstream::in | fstream::binary);
    if (!ifs.good()) {
        LOG_ERROR(("Open word2vec file failed, filename: ")(filename));
        return false;
    }
    ScopedFstream<ifstream> scoped_ifs(ifs);
    string line;
    vector<string> flds;
    int_t nwords, nvecds;
    int_t linu = 0;
    getline(ifs, line);
    ++linu;
    StringHandler::RightTrimString(line, line, "\n\r");
    StringHandler::SplitString(line, flds);
    nwords = atoi(flds[0].c_str());

    nvecds = atoi(flds[1].c_str());
    if (nvecds != nfd[0]) {
        LOG_ERROR(("word2vec dim is not equal to nfd[0]"));
        return false;
    }
    delete[] feature_table[0];
    feature_table[0] = new real_t[(nwords + 1) * nvecds];
    int_t wid = 0;
    feat_dicts[0].clear();
    feat_dicts[0].insert(StrToIntMap::value_type(NIL, wid++));
    int_t i;
    for (i = 0; i < nvecds; ++i) { feature_table[0][i] = 0; }
    do {
        getline(ifs, line);
        ++linu;
        StringHandler::RightTrimString(line, line, " \n\r");
        if (line.empty()) {
            if (!ifs.good()) { break; }
            else             { continue; }
        }
        StringHandler::SplitString(line, flds, " ");
        if (nvecds + 1 != static_cast<int_t>(flds.size())) {
            LOG_ERROR(("Field number not right in file: ")(filename)
                (", line number: ")(linu)("content: ")(line));
            return false;
        }
        if (feat_dicts[0].insert(StrToIntMap::value_type(flds[0], wid)).second){
            for (i = 0; i < nvecds; ++i) {
                feature_table[0][wid * nvecds + i] = atof(flds[i + 1].c_str());
            }
            ++wid;
        }
    } while (ifs.good() && wid <= nwords);
    if (wid != nwords + 1) {
        LOG_WARN(("WordNum is not right in word2vec file: ")(filename));
    }
    LOG_INFO(("Finish load Word2vec. WordNum = ")(wid));
    isUsingWord2vec = 1;
    return true;
}

void FdcnnNet::logInfoConfig()
{
    int_t i = 0;
    ostringstream oss;
    LOG_INFO(("dropout = ")(dropout_prob));
    LOG_INFO(("nif = ")(nif));
    oss.clear();
    oss.str("");
    oss << "nfd =";
    for (i = 0; i < nif; ++i) { oss << ' ' << nfd[i]; }
    LOG_INFO((oss.str()));
    LOG_INFO(("nsr = ")(nsr));
    LOG_INFO(("ncl = ")(ncl));
    oss.clear();
    oss.str("");
    oss << "scw =";
    for (i = 0; i < ncl; ++i) { oss << ' ' << scw[i]; }
    LOG_INFO((oss.str()));
    oss.clear();
    oss.str("");
    oss << "ncr =";
    for (i = 0; i < ncl; ++i) { oss << ' ' << ncr[i]; }
    LOG_INFO((oss.str()));
    oss.clear();
    oss.str("");
    oss << "ncm =";
    for (i = 0; i < ncl; ++i) { oss << ' ' << ncm[i]; }
    LOG_INFO((oss.str()));
    LOG_INFO(("Ktop = ")(Ktop));
    LOG_INFO(("nhl = ")(nhl));
    oss.clear();
    oss.str("");
    oss << "nhr =";
    for (i = 0; i < nhl; ++i) { oss << ' ' << nhr[i]; }
    LOG_INFO((oss.str()));
}

bool FdcnnNet::LoadModel(const char * filename){
    FILE * model_file = fopen(filename, "rb");
    if (!model_file) {
        LOG_ERROR(("Open model file failed, filename: ")(filename));
        return false;
    }
    ScopedFile scoped_file(model_file);
    LOG_INFO(("Start loading model."));
    freeConfigMemory();
    int_t i1, i2, i3, i4, i5, tmp, csize, dictsize, index, size;
    char fld[MAX_WORD_LEN];
    fscanf(model_file, "dropout = %lf\n", &dropout_prob);
    fscanf(model_file, "nif = %d\n", &nif);
    nfd = new int_t[nif];
    fscanf(model_file, "nfd = ");
    for (i1 = 0; i1 < nif; ++i1) { fscanf(model_file, "%d ", &nfd[i1]); }
    fscanf(model_file, "\nnwd = %d\n", &nsr);
    fscanf(model_file, "ncl = %d\n", &ncl);
    scw = new int_t[ncl];
    ncr = new int_t[ncl];
    ncm = new int_t[ncl];
    fscanf(model_file, "scw = ");
    for (i1 = 0; i1 < ncl; ++i1) { fscanf(model_file, "%d ", &scw[i1]); }
    fscanf(model_file, "\nncr = ");
    for (i1 = 0; i1 < ncl; ++i1) { fscanf(model_file, "%d ", &ncr[i1]); }
    fscanf(model_file, "\nncm = ");
    for (i1 = 0; i1 < ncl; ++i1) { fscanf(model_file, "%d ", &ncm[i1]); }
    fscanf(model_file, "\nKtop = %d", &Ktop);
    fscanf(model_file, "\nnhl = %d", &nhl);
    nhr = new int_t[nhl];
    fscanf(model_file, "\nnhr = ");
    for (i1 = 0; i1 < nhl; ++i1) { fscanf(model_file, "%d ", &nhr[i1]); }
    fscanf(model_file, "\n");
    fscanf(model_file, "\n");
    logInfoConfig();
    feat_dicts.clear();
    feat_dicts.resize(nif);
    label_dict.clear();
    for (i1 = 0; i1 < nif; ++i1){
        fscanf(model_file, "feat_dicts[%d]\tsize = %d\n", &tmp, &dictsize);
        for (i2 = 0; i2 < dictsize; ++i2) {
            fscanf(model_file, "%s\t%d\n", fld, &index);
            feat_dicts[i1].insert(StrToIntMap::value_type(string(fld), index));
        }
        fscanf(model_file, "\n");
    }
    fscanf(model_file, "label_dict\tsize = %d\n", &dictsize);
    for (i2 = 0; i2 < dictsize; ++i2) {
        fscanf(model_file, "%s\t%d\n", fld, &index);
        label_dict.insert(StrToIntMap::value_type(string(fld), index));
    }
    fscanf(model_file, "\n");
    nor = static_cast<int_t>(label_dict.size());
    allocWeightMemory();
    allocFeatureMemory();
    for (i1 = 0; i1 < nif; ++i1){
        fscanf(model_file, "feature_table[%d]\tsize = %d * %d\n", &index,
               &dictsize, &nfd[i1]);
        size = static_cast<int_t>(feat_dicts[i1].size());
        for (i2 = 0; i2 < size; ++i2){
            for (i3 = 0; i3 < nfd[i1]; ++i3) {
                fscanf(model_file, "%lf ",
                       &feature_table[i1][i2 * nfd[i1] + i3]);
            }
            fscanf(model_file, "\n");
        }
        fscanf(model_file, "\n");
    }
    csize = nsr * scw[0] + 1;
    for (i2 = 0; i2 < ncm[0]; ++i2) {
        fscanf(model_file, "MM[0][%d][0]\t size = %d * %d\n", &index, &ncr[0],
               &csize);
        for (i4 = 0; i4 < ncr[0]; ++i4){
            for (i5 = 0; i5 < csize; ++i5) {
                fscanf(model_file, "%lf ", &MM[0][i2][i4 * csize + i5]);
            }
            fscanf(model_file, "\n");
        }
        fscanf(model_file, "\n");
    }
    for (i1 = 1, tmp = 0; i1 < ncl; ++i1, ++tmp) {
        csize = ncr[tmp] * scw[i1] + 1;
        for (i2 = 0; i2 < ncm[i1]; ++i2) {
            for (i3 = 0; i3 < ncm[tmp]; ++i3){
                fscanf(model_file, "MM[%d][%d][%d]\t size = %d * %d\n", &index,
                       &index, &index, &ncr[i1], &csize);
                for (i4 = 0; i4 < ncr[i1]; ++i4){
                    for (i5 = 0; i5 < csize; ++i5){
                        fscanf(model_file, "%lf ", 
                               &MM[i1][i2 * ncm[tmp] + i3][i4 * csize + i5]);
                    }
                    fscanf(model_file, "\n");
                }
                fscanf(model_file, "\n");
            }
        }
    }
    csize = ncm[ncl - 1] * Ktop * ncr[ncl - 1] + 1;
    fscanf(model_file, "WM[0]\tsize = %d * %d\n", &nhr[0], &csize);
    for (i2 = 0; i2 < nhr[0]; ++i2){
        for (i3 = 0; i3 < csize; ++i3) {
            fscanf(model_file, "%lf ", &WM[0][i2 * csize + i3]);
        }
        fscanf(model_file, "\n");
    }
    fscanf(model_file, "\n");
    for (i1 = 1, tmp = 0; i1 < nhl; ++i1, ++tmp) {
        csize = nhr[tmp] + 1;
        fscanf(model_file, "WM[%d]\t size = %d * %d\n", &index, &nhr[i1],
               &csize);
        for (i2 = 0; i2 < nhr[i1]; ++i2) {
            for (i3 = 0; i3 < csize; ++i3) {
                fscanf(model_file, "%lf ", &WM[i1][i2 * csize + i3]);
            }
            fscanf(model_file, "\n");
        }
        fscanf(model_file, "\n");
    }
    csize = nhr[nhl - 1] + 1;
    fscanf(model_file, "U\tsize = %d * %d\n", &nor, &csize);
    for (i1 = 0; i1 < nor; ++i1){
        for (i2 = 0; i2 < csize; ++i2) {
            fscanf(model_file, "%lf ", &U[i1 * csize + i2]);
        }
        fscanf(model_file, "\n");
    }
    fscanf(model_file, "\n");
    makeIndex2label();
    LOG_INFO(("Finish loading model."));
    return true;
}

bool FdcnnNet::SaveModel(const char * filename)
{
    FILE * model_file = fopen(filename, "wb");
    if (!model_file) {
        LOG_ERROR(("Open model file failed, filename: ")(filename));
        return false;
    }
    ScopedFile scoped_file(model_file);
    LOG_INFO(("Start saving model."));
    int_t i1, i2, i3, i4, i5, tmp, csize, size;
    fprintf(model_file, "dropout = %f\n", dropout_prob);
    fprintf(model_file, "nif = %d\n", nif);
    fprintf(model_file, "nfd = ");
    for (i1 = 0; i1 < nif; ++i1) { fprintf(model_file, "%d ", nfd[i1]); }
    fprintf(model_file, "\nnwd = %d\n", nsr);
    fprintf(model_file, "ncl = %d\n", ncl);
    fprintf(model_file, "scw = ");
    for (i1 = 0; i1 < ncl; ++i1) { fprintf(model_file, "%d ", scw[i1]); }
    fprintf(model_file, "\nncr = ");
    for (i1 = 0; i1 < ncl; ++i1) { fprintf(model_file, "%d ", ncr[i1]); }
    fprintf(model_file, "\nncm = ");
    for (i1 = 0; i1 < ncl; ++i1) { fprintf(model_file, "%d ", ncm[i1]); }
    fprintf(model_file, "\nKtop = %d", Ktop);
    fprintf(model_file, "\nnhl = %d", nhl);
    fprintf(model_file, "\nnhr = ");
    for (i1 = 0; i1 < nhl; ++i1) { fprintf(model_file, "%d ", nhr[i1]); }
    fprintf(model_file, "\n");
    fprintf(model_file, "\n");

    for (i1 = 0; i1 < nif; ++i1) {
        fprintf(model_file, "feat_dicts[%d]\tsize = %d\n", i1,
            static_cast<int_t>(feat_dicts[i1].size()));
        for (StrToIntMap::iterator it = feat_dicts[i1].begin();
            it != feat_dicts[i1].end(); ++it) {
            fprintf(model_file, "%s\t%d\n", it->first.c_str(), it->second);
        }
        fprintf(model_file, "\n");
    }
    fprintf(model_file, "label_dict\tsize = %d\n",
        static_cast<int_t>(label_dict.size()));
    for (StrToIntMap::iterator it = label_dict.begin();
        it != label_dict.end(); ++it) {
        fprintf(model_file, "%s\t%d\n", it->first.c_str(), it->second);
    }
    fprintf(model_file, "\n");


    for (i1 = 0; i1 < nif; ++i1){
        fprintf(model_file, "feature_table[%d]\tsize = %d * %d\n", i1,
            static_cast<int_t>(feat_dicts[i1].size()), nfd[i1]);
        size = static_cast<int_t>(feat_dicts[i1].size());
        for (i2 = 0; i2 < size; ++i2) {
            for (i3 = 0; i3 < nfd[i1]; ++i3) {
                fprintf(model_file, "%lf ", 
                        feature_table[i1][i2 * nfd[i1] + i3]);
            }
            fprintf(model_file, "\n");
        }
        fprintf(model_file, "\n");
    }
    csize = nsr * scw[0] + 1;
    for (i2 = 0; i2 < ncm[0]; ++i2) {
        fprintf(model_file, "MM[0][%d][0]\t size = %d * %d\n", i2,
                ncr[0], csize);
        for (i4 = 0; i4 < ncr[0]; ++i4){
            for (i5 = 0; i5 < csize; ++i5) {
                fprintf(model_file, "%lf ", MM[0][i2][i4 * csize + i5]);
            }
            fprintf(model_file, "\n");
        }
        fprintf(model_file, "\n");
    }
    for (i1 = 1, tmp = 0; i1 < ncl; ++i1, ++tmp) {
        csize = ncr[tmp] * scw[i1] + 1;
        for (i2 = 0; i2 < ncm[i1]; ++i2)  {
            for (i3 = 0; i3 < ncm[tmp]; ++i3){
                fprintf(model_file, "MM[%d][%d][%d]\t size = %d * %d\n", 
                        i1, i2, i3, ncr[i1], csize);
                for (i4 = 0; i4 < ncr[i1]; ++i4){
                    for (i5 = 0; i5 < csize; ++i5) {
                        fprintf(model_file, "%lf ",
                                MM[i1][i2 * ncm[tmp] + i3][i4 * csize + i5]);
                    }
                    fprintf(model_file, "\n");
                }
                fprintf(model_file, "\n");
            }
        }
    }
    csize = ncm[ncl - 1] * Ktop * ncr[ncl - 1] + 1;
    fprintf(model_file, "WM[0]\tsize = %d * %d\n", nhr[0], csize);
    for (i2 = 0; i2 < nhr[0]; ++i2){
        for (i3 = 0; i3 < csize; ++i3) {
            fprintf(model_file, "%lf ", WM[0][i2 * csize + i3]);
        }
        fprintf(model_file, "\n");
    }
    fprintf(model_file, "\n");
    for (i1 = 1, tmp = 0; i1 < nhl; ++i1, ++tmp) {
        csize = nhr[tmp] + 1;
        fprintf(model_file, "WM[%d]\t size = %d * %d\n", i1, nhr[i1], csize);
        for (i2 = 0; i2 < nhr[i1]; ++i2){
            for (i3 = 0; i3 < csize; ++i3) {
                fprintf(model_file, "%lf ", WM[i1][i2 * csize + i3]);
            }
            fprintf(model_file, "\n");
        }
        fprintf(model_file, "\n");
    }
    csize = nhr[nhl - 1] + 1;
    fprintf(model_file, "U\tsize = %d * %d\n", nor, csize);
    for (i1 = 0; i1 < nor; ++i1){
        for (i2 = 0; i2 < csize; ++i2) {
            fprintf(model_file, "%lf ", U[i1 * csize + i2]);
        }
        fprintf(model_file, "\n");
    }
    fprintf(model_file, "\n");
    LOG_INFO(("Finish saving model."));
    return true;
}

bool FdcnnNet::SaveModelBinary(const char * filename) {
    FILE * model_file = fopen(filename, "wb");
    if (!model_file) {
        LOG_ERROR(("Open model file failed, filename: ")(filename));
        return false;
    }
    ScopedFile scoped_file(model_file);
    LOG_INFO(("Start saving model."));
    int_t i1, i2, i3, tmp, csize;
    fwrite(&dropout_prob, sizeof(real_t), 1, model_file);
    fwrite(&nif, sizeof(int_t), 1, model_file);
    fwrite(nfd, sizeof(int_t), nif, model_file);
    fwrite(&nsr, sizeof(int_t), 1, model_file);
    fwrite(&ncl, sizeof(int_t), 1, model_file);
    fwrite(scw, sizeof(int_t), ncl, model_file);
    fwrite(ncr, sizeof(int_t), ncl, model_file);
    fwrite(ncm, sizeof(int_t), ncl, model_file);
    fwrite(&Ktop, sizeof(int_t), 1, model_file);
    fwrite(&nhl, sizeof(int_t), 1, model_file);
    fwrite(nhr, sizeof(int_t), nhl, model_file);
    for (i1 = 0; i1 < nif; ++i1){
        csize = static_cast<int_t>(feat_dicts[i1].size());
        fwrite(&csize, sizeof(int_t), 1, model_file);
        for (StrToIntMap::iterator it = feat_dicts[i1].begin();
            it != feat_dicts[i1].end(); ++it) {
            fprintf(model_file, "%s\n", it->first.c_str());
            csize = it->second;
            fwrite(&csize, sizeof(int_t), 1, model_file);
        }
    }

    csize = static_cast<int_t>(label_dict.size());
    fwrite(&csize, sizeof(int_t), 1, model_file);
    for (StrToIntMap::iterator it = label_dict.begin();
        it != label_dict.end(); ++it) {
        fprintf(model_file, "%s\n", it->first.c_str());
        csize = it->second;
        fwrite(&csize, sizeof(int_t), 1, model_file);
    }

    for (i1 = 0; i1 < nif; ++i1) {
        csize = static_cast<int_t>(feat_dicts[i1].size()) * nfd[i1];
        csize = fwrite(feature_table[i1], sizeof(real_t), csize, model_file);
    }
    csize = nsr * scw[0] + 1;
    for (i2 = 0; i2 < ncm[0]; ++i2) {
        fwrite(MM[0][i2], sizeof(real_t), ncr[0] * csize, model_file);
    }
    for (i1 = 1, tmp = 0; i1 < ncl; ++i1, ++tmp){
        csize = ncr[tmp] * scw[i1] + 1;
        for (i2 = 0; i2 < ncm[i1]; ++i2) {
            for (i3 = 0; i3 < ncm[tmp]; ++i3) {
                fwrite(MM[i1][i2 * ncm[tmp] + i3], sizeof(real_t),
                       ncr[i1] * csize, model_file);
            }
        }
    }
    fwrite(WM[0], sizeof(real_t), 
           nhr[0] * (ncm[ncl - 1] * Ktop * ncr[ncl - 1] + 1), model_file);
    for (i1 = 1, tmp = 0; i1 < nhl; ++i1, ++tmp) {
        fwrite(WM[i1], sizeof(real_t), nhr[i1] * (nhr[tmp] + 1), model_file);
    }
    fwrite(U, sizeof(real_t), nor * (nhr[nhl - 1] + 1), model_file);
    LOG_INFO(("Finish saving model."));
    return true;
}

bool FdcnnNet::LoadModelBinary(const char * filename){
    FILE * model_file = fopen(filename, "rb");
    if (!model_file) {
        LOG_ERROR(("Open model file failed, filename: ")(filename));
        return false;
    }
    ScopedFile scoped_file(model_file);
    LOG_INFO(("Start loading model."));
    freeConfigMemory();
    int_t i1, i2, i3, len, tmp, csize, index;
    char fld[MAX_WORD_LEN + 2];
    fread(&dropout_prob, sizeof(real_t), 1, model_file);
    fread(&nif, sizeof(int_t), 1, model_file);
    nfd = new int_t [nif];
    fread(nfd, sizeof(int_t), nif, model_file);
    fread(&nsr, sizeof(int_t), 1, model_file);
    fread(&ncl, sizeof(int_t), 1, model_file);
    scw = new int_t[ncl];
    ncr = new int_t[ncl];
    ncm = new int_t[ncl];
    fread(scw, sizeof(int_t), ncl, model_file);
    fread(ncr, sizeof(int_t), ncl, model_file);
    fread(ncm, sizeof(int_t), ncl, model_file);
    fread(&Ktop, sizeof(int_t), 1, model_file);
    fread(&nhl, sizeof(int_t), 1, model_file);
    nhr = new int_t[nhl];
    fread(nhr, sizeof(int_t), nhl, model_file);

    logInfoConfig();
    feat_dicts.clear();
    feat_dicts.resize(nif);
    label_dict.clear();
    for (i1 = 0; i1 < nif; ++i1){
        fread(&csize, sizeof(int_t), 1, model_file);
        for (i2 = 0; i2 < csize; ++i2){
            fgets(fld, MAX_WORD_LEN + 2, model_file);
            len = strlen(fld);
            while ((fld[len - 1] == '\n' || fld[len - 1] == '\r') && len > 0) {
                fld[--len] = 0;
            }
            fread(&index, sizeof(int_t), 1, model_file);
            feat_dicts[i1].insert(StrToIntMap::value_type(string(fld), index));
        }
    }
    fread(&csize, sizeof(int_t), 1, model_file);
    for (i2 = 0; i2 < csize; ++i2){
        fgets(fld, MAX_WORD_LEN + 2, model_file);
        len = strlen(fld);
        while ((fld[len - 1] == '\n' || fld[len - 1] == '\r') && len > 0) {
            fld[--len] = 0;
        }
        fread(&index, sizeof(int_t), 1, model_file);
        label_dict.insert(StrToIntMap::value_type(string(fld), index));
    }
    nor = static_cast<int_t>(label_dict.size());
    allocWeightMemory();
    allocFeatureMemory();
    for (i1 = 0; i1 < nif; ++i1)
    {
        csize = static_cast<int_t>(feat_dicts[i1].size()) * nfd[i1];
        csize = fread(feature_table[i1], sizeof(real_t), csize, model_file);
    }
    csize = nsr * scw[0] + 1;
    for (i2 = 0; i2 < ncm[0]; ++i2) {
        fread(MM[0][i2], sizeof(real_t), ncr[0] * csize, model_file);
    }
    for (i1 = 1, tmp = 0; i1 < ncl; ++i1, ++tmp){
        csize = ncr[tmp] * scw[i1] + 1;
        for (i2 = 0; i2 < ncm[i1]; ++i2) {
            for (i3 = 0; i3 < ncm[tmp]; ++i3) {
                fread(MM[i1][i2 * ncm[tmp] + i3], sizeof(real_t),
                      ncr[i1] * csize, model_file);
            }
        }
    }
    fread(WM[0], sizeof(real_t),
          nhr[0] * (ncm[ncl - 1] * Ktop * ncr[ncl - 1] + 1), model_file);
    for (i1 = 1, tmp = 0; i1 < nhl; ++i1, ++tmp) {
        fread(WM[i1], sizeof(real_t), nhr[i1] * (nhr[tmp] + 1), model_file);
    }
    fread(U, sizeof(real_t), nor * (nhr[nhl - 1] + 1), model_file);
    makeIndex2label();
    LOG_INFO(("Finish loading model."));
    return true;
}

void FdcnnNet::allocTrainRunMemory()
{
    LOG_DEBUG(("Enter AllocRunMemory"));
    freeRunMemory();
    int_t i1, i2, i3, tmp1, tmp2, tmp3, tmp4;
    //alloc menory
    diff_MM     = new real_t***[thrd_capacity]; 
    diff_WM     = new real_t** [thrd_capacity];
    diff_U = new real_t*[thrd_capacity];
    ncc = new int_t*[thrd_capacity];
    ndk = new int_t*[thrd_capacity];
    conv_M = new real_t***[thrd_capacity];
    diff_conv_M = new real_t***[thrd_capacity];
    dropout_conv_mask_M = new int_t***[thrd_capacity];
    pool_V = new int_t***[thrd_capacity];
    pool_M = new real_t ***[thrd_capacity];
    diff_pool_M = new real_t ***[thrd_capacity];
    hid_V = new real_t**[thrd_capacity];
    diff_hid_V = new real_t**[thrd_capacity];
    dropout_hid_mask_V = new int_t**[thrd_capacity];
    sen_M = new real_t*[thrd_capacity];
    diff_sen_M = new real_t*[thrd_capacity];
    out_V = new real_t*[thrd_capacity];
    diff_out_V = new real_t*[thrd_capacity];
    lsen = new int_t[thrd_capacity];

    for (i1 = 0; i1 < thrd_capacity; ++i1) {
        //1
        sen_M[i1] = new real_t[nsr * MAX_SEN_LEN];
        diff_sen_M[i1] = new real_t[nsr * MAX_SEN_LEN];
        out_V[i1] = new real_t[nor];
        diff_out_V[i1] = new real_t[nor];

        //2
        ncc[i1] = new int_t[ncl];
        ndk[i1] = new int_t[ncl];
        conv_M[i1] = new real_t **[ncl];
        diff_conv_M[i1] = new real_t **[ncl];
        dropout_conv_mask_M[i1] = new int_t **[ncl];
        pool_V[i1] = new int_t**[ncl];
        pool_M[i1] = new real_t **[ncl];
        diff_pool_M[i1] = new real_t **[ncl];
        diff_MM[i1] = new real_t**[ncl];
        for (i2 = 0; i2 < ncl; ++i2) {
            conv_M[i1][i2] = new real_t*[ncm[i2]];
            diff_conv_M[i1][i2] = new real_t*[ncm[i2]];
            dropout_conv_mask_M[i1][i2] = new int_t *[ncm[i2]];
            pool_V[i1][i2] = new int_t *[ncm[i2]];
            pool_M[i1][i2] = new real_t *[ncm[i2]];
            diff_pool_M[i1][i2] = new real_t *[ncm[i2]];
            if (i2 == 0) {
                tmp1 = ncr[i2] * (scw[i2] + MAX_SEN_LEN - 1);
                tmp3 = ncm[i2];
                tmp4 = ncr[i2] * (scw[i2] * nsr + 1);
            } else {
                tmp1 = ncr[i2] * (scw[i2] + dynamicK(MAX_SEN_LEN, i2 - 1) - 1);
                tmp3 = ncm[i2 - 1] * ncm[i2];
                tmp4 = ncr[i2] * (scw[i2] * ncr[i2 - 1] + 1);
            }
            tmp2 = dynamicK(MAX_SEN_LEN, i2);

            diff_MM[i1][i2] = new real_t*[tmp3];
            for (i3 = 0; i3 < tmp3; ++i3) {
                diff_MM[i1][i2][i3] = new real_t[tmp4];
            }
            for (i3 = 0; i3 < ncm[i2]; ++i3) {
                conv_M[i1][i2][i3] = new real_t[tmp1];
                diff_conv_M[i1][i2][i3] = new real_t[tmp1];
                dropout_conv_mask_M[i1][i2][i3] = new int_t[tmp1];
                pool_V[i1][i2][i3] = new int_t[tmp2];
                pool_M[i1][i2][i3] = new real_t[tmp2 * ncr[i2]];
                diff_pool_M[i1][i2][i3] = new real_t[tmp2 * ncr[i2]];
            }
        }

        //3
        diff_WM[i1]     = new real_t*[nhl];
        hid_V[i1]       = new real_t*[nhl];
        diff_hid_V[i1]  = new real_t*[nhl];
        dropout_hid_mask_V[i1] = new int_t*[nhl];
        for (i2 = 0; i2 < nhl; ++i2) {
            hid_V[i1][i2] = new real_t[nhr[i2]];
            diff_hid_V[i1][i2] = new real_t[nhr[i2]];
            dropout_hid_mask_V[i1][i2] = new int_t[nhr[i2]];
            if (0 == i2) {
                tmp1 = nhr[i2] * (Ktop * ncr[ncl - 1] * ncm[ncl - 1] + 1);
            } else {
                tmp1 = nhr[i2] * (nhr[i2 - 1] + 1);
            }
            diff_WM[i1][i2] = new real_t[tmp1];
        }
        //4
        diff_U[i1] = new real_t[nor * (nhr[nhl - 1] + 1)];
    }
    LOG_DEBUG(("Leave AllocRunMemory"));
}

void FdcnnNet::allocPredictRunMemory()
{
    LOG_DEBUG(("Enter allocPredictForwardMemory"));
    freeRunMemory();
    int_t i1, i2, i3, tmp1, tmp2;
    //alloc menory

    ncc = new int_t*[thrd_capacity];
    ndk = new int_t*[thrd_capacity];
    conv_M = new real_t***[thrd_capacity];
    pool_V = new int_t***[thrd_capacity];
    pool_M = new real_t ***[thrd_capacity];
    hid_V = new real_t**[thrd_capacity];
    sen_M = new real_t*[thrd_capacity];
    out_V = new real_t*[thrd_capacity];
    lsen = new int_t[thrd_capacity];

    for (i1 = 0; i1 < thrd_capacity; ++i1) {
        //1
        sen_M[i1] = new real_t[nsr * MAX_SEN_LEN];
        out_V[i1] = new real_t[nor];

        //2
        ncc[i1] = new int_t[ncl];
        ndk[i1] = new int_t[ncl];
        conv_M[i1] = new real_t **[ncl];
        pool_V[i1] = new int_t**[ncl];
        pool_M[i1] = new real_t **[ncl];
        for (i2 = 0; i2 < ncl; ++i2) {
            conv_M[i1][i2] = new real_t*[ncm[i2]];
            pool_V[i1][i2] = new int_t *[ncm[i2]];
            pool_M[i1][i2] = new real_t *[ncm[i2]];
            if (i2 == 0) {
                tmp1 = ncr[i2] * (scw[i2] + MAX_SEN_LEN - 1);
            }
            else {
                tmp1 = ncr[i2] * (scw[i2] + dynamicK(MAX_SEN_LEN, i2 - 1) - 1);
            }
            tmp2 = dynamicK(MAX_SEN_LEN, i2);

            for (i3 = 0; i3 < ncm[i2]; ++i3) {
                conv_M[i1][i2][i3] = new real_t[tmp1];
                pool_V[i1][i2][i3] = new int_t[tmp2];
                pool_M[i1][i2][i3] = new real_t[tmp2 * ncr[i2]];
            }
        }
        //3
        hid_V[i1] = new real_t*[nhl];
        for (i2 = 0; i2 < nhl; ++i2) { hid_V[i1][i2] = new real_t[nhr[i2]]; }
    }
    busy_map.assign(thrd_capacity, 0);
    busy_num = 0;
    LOG_DEBUG(("Leave allocPredictForwardMemory"));
}

int_t FdcnnNet::registerRunRes()
{
    LOG_DEBUG(("Enter RegisterRunRes"));
    while (true) {
        do {
            ScopedLock scoped_lock(busy_num_mutex);
            if (busy_num < thrd_capacity) {
                ++busy_num;
                for (int_t i = 0; i < thrd_capacity; ++i) {
                    if (!busy_map[i]) {
                        LOG_DEBUG(("Leave RegisterRunRes, res_id: ")(i));
                        return i;
                    }
                }
                LOG_ERROR(("GetResourceFalied, busy_num = ")(busy_num));
                return -1;
            }
        } while (false);
        usleep(USLEEP_INTERVAL);
    }
}

void FdcnnNet::releaseRunRes(int_t id)
{
    LOG_DEBUG(("Enter ReleaseRunRes"));
    do {
        ScopedLock scoped_lock(busy_num_mutex);
        --busy_num;
        busy_map[id] = 0;
    } while (false);
    LOG_DEBUG(("Leave ReleaseRunRes, res_id: ")(id));
}

void FdcnnNet::updateParams(int_t thrd_id)
{
    LOG_DEBUG(("Enter UpdateParams"));
    int_t i1, i2, i3, tmp1, tmp2;
    for (i1 = 0; i1 < ncl; ++i1) {
        if (i1 == 0) {
            tmp1 = ncm[i1];
            tmp2 = ncr[i1] * (scw[i1] * nsr + 1);
        }
        else {
            tmp1 = ncm[i1 - 1] * ncm[i1];
            tmp2 = ncr[i1] * (scw[i1] * ncr[i1 - 1] + 1);
        }
        for (i2 = 0; i2 < tmp1; ++i2) {
            for (i3 = 0; i3 < tmp2; ++i3) {
                MM[i1][i2][i3] -= diff_MM[thrd_id][i1][i2][i3] * alpha
                    + lambda * MM[i1][i2][i3];
            }
        }
    }
    for (i1 = 0; i1 < nhl; ++i1) {
        if (0 == i1) {
            tmp1 = nhr[i1] * (Ktop * ncr[ncl - 1] * ncm[ncl - 1] + 1);
        }
        else {
            tmp1 = nhr[i1] * (nhr[i1 - 1] + 1);
        }
        for (i2 = 0; i2 < tmp1; ++i2) {
            WM[i1][i2] -= diff_WM[thrd_id][i1][i2] * alpha
                + lambda * WM[i1][i2];
        }
    }
    tmp1 = nor * (nhr[nhl - 1] + 1);
    for (i1 = 0; i1 < tmp1; ++i1) {
        U[i1] -= diff_U[thrd_id][i1] * alpha + lambda * U[i1];
    }
    LOG_DEBUG(("Leave UpdateParams"));
}

void FdcnnNet::resetDiff(int_t thrd_id)
{
    LOG_DEBUG(("Enter ResetDiff"));
    int_t i1, i2, i3, tmp1, tmp2;
    for (i1 = 0; i1 < ncl; ++i1) {
        if (i1 == 0) {
            tmp1 = ncm[i1];
            tmp2 = ncr[i1] * (scw[i1] * nsr + 1);
        }
        else {
            tmp1 = ncm[i1 - 1] * ncm[i1];
            tmp2 = ncr[i1] * (scw[i1] * ncr[i1 - 1] + 1);
        }
        for (i2 = 0; i2 < tmp1; ++i2) {
            for (i3 = 0; i3 < tmp2; ++i3) {
                diff_MM[thrd_id][i1][i2][i3] = 0;
            }
        }
    }
    for (i1 = 0; i1 < nhl; ++i1) {
        if (0 == i1) {
            tmp1 = nhr[i1] * (Ktop * ncr[ncl - 1] * ncm[ncl - 1] + 1);
        }
        else {
            tmp1 = nhr[i1] * (nhr[i1 - 1] + 1);
        }
        for (i2 = 0; i2 < tmp1; ++i2) {
            diff_WM[thrd_id][i1][i2] = 0;
        }
    }
    tmp1 = nor * (nhr[nhl - 1] + 1);
    for (i1 = 0; i1 < tmp1; ++i1) {
        diff_U[thrd_id][i1] = 0;
    }
    LOG_DEBUG(("Leave ResetDiff"));
}

void FdcnnNet::freeRunMemory()
{
    LOG_DEBUG(("Enter FreeRunMemory"));
    int_t i1, i2, i3, tmp;
    if (diff_MM) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            for (i2 = 0; i2 < ncl; ++i2) {
                if (i2 == 0) {
                    tmp = ncm[i2];
                }
                else {
                    tmp = ncm[i2 - 1] * ncm[i2];
                }
                for (i3 = 0; i3 < tmp; ++i3) {
                    delete[] diff_MM[i1][i2][i3];
                }
                delete[] diff_MM[i1][i2];
            }
            delete[] diff_MM[i1];
        }
        delete[] diff_MM;
        diff_MM = 0;
    }
    if (diff_WM) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            for (i2 = 0; i2 < nhl; ++i2) {
                delete[] diff_WM[i1][i2];
            }
            delete[] diff_WM[i1];
        }
        delete[] diff_WM;
        diff_WM = 0;
    }
    if (diff_U) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            delete[] diff_U[i1];
        }
        delete[] diff_U;
        diff_U = 0;
    }
    if (sen_M) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            delete[] sen_M[i1];
        }
        delete[] sen_M;
        sen_M = 0;
    }
    if (diff_sen_M) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            delete[] diff_sen_M[i1];
        }
        delete[] diff_sen_M;
        diff_sen_M = 0;
    }
    if (lsen) {
        delete[] lsen;
        lsen = 0;
    }
    if (ncc) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            delete[] ncc[i1];
        }
        delete[] ncc;
        ncc = 0;
    }
    if (ndk) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            delete[] ndk[i1];
        }
        delete[] ndk;
        ndk = 0;
    }
    if (conv_M) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            for (i2 = 0; i2 < ncl; ++i2) {
                for (i3 = 0; i3 < ncm[i2]; ++i3) {
                    delete[] conv_M[i1][i2][i3];
                }
                delete[] conv_M[i1][i2];
            }
            delete[] conv_M[i1];
        }
        delete[] conv_M;
        conv_M = 0;
    }
    if (diff_conv_M) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            for (i2 = 0; i2 < ncl; ++i2) {
                for (i3 = 0; i3 < ncm[i2]; ++i3) {
                    delete[] diff_conv_M[i1][i2][i3];
                }
                delete[] diff_conv_M[i1][i2];
            }
            delete[] diff_conv_M[i1];
        }
        delete[] diff_conv_M;
        diff_conv_M = 0;
    }
    if (dropout_conv_mask_M) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            for (i2 = 0; i2 < ncl; ++i2) {
                for (i3 = 0; i3 < ncm[i2]; ++i3) {
                    delete[] dropout_conv_mask_M[i1][i2][i3];
                }
                delete[] dropout_conv_mask_M[i1][i2];
            }
            delete[] dropout_conv_mask_M[i1];
        }
        delete[] dropout_conv_mask_M;
        dropout_conv_mask_M = 0;
    }
    if (pool_V) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            for (i2 = 0; i2 < ncl; ++i2) {
                for (i3 = 0; i3 < ncm[i2]; ++i3) {
                    delete[] pool_V[i1][i2][i3];
                }
                delete[] pool_V[i1][i2];
            }
            delete[] pool_V[i1];
        }
        delete[] pool_V;
        pool_V = 0;
    }
    if (pool_M) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            for (i2 = 0; i2 < ncl; ++i2) {
                for (i3 = 0; i3 < ncm[i2]; ++i3) {
                    delete[] pool_M[i1][i2][i3];
                }
                delete[] pool_M[i1][i2];
            }
            delete[] pool_M[i1];
        }
        delete[] pool_M;
        pool_M = 0;
    }
    if (diff_pool_M) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            for (i2 = 0; i2 < ncl; ++i2) {
                for (i3 = 0; i3 < ncm[i2]; ++i3) {
                    delete[] diff_pool_M[i1][i2][i3];
                }
                delete[] diff_pool_M[i1][i2];
            }
            delete[] diff_pool_M[i1];
        }
        delete[] diff_pool_M;
        diff_pool_M = 0;
    }
    if (hid_V) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            for (i2 = 0; i2 < nhl; ++i2) {
                delete[] hid_V[i1][i2];
            }
            delete[] hid_V[i1];
        }
        delete[] hid_V;
        hid_V = 0;
    }
    if (diff_hid_V) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            for (i2 = 0; i2 < nhl; ++i2) {
                delete[] diff_hid_V[i1][i2];
            }
            delete[] diff_hid_V[i1];
        }
        delete[] diff_hid_V;
        diff_hid_V = 0;
    }
    if (dropout_hid_mask_V) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            for (i2 = 0; i2 < nhl; ++i2) {
                delete[] dropout_hid_mask_V[i1][i2];
            }
            delete[] dropout_hid_mask_V[i1];
        }
        delete[] dropout_hid_mask_V;
        dropout_hid_mask_V = 0;
    }
    if (out_V) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            delete[] out_V[i1];
        }
        delete[] out_V;
        out_V = 0;
    }
    if (diff_out_V) {
        for (i1 = 0; i1 < thrd_capacity; ++i1) {
            delete[] diff_out_V[i1];
        }
        delete[] diff_out_V;
        diff_out_V = 0;
    }

    LOG_DEBUG(("Leave FreeRunMemory"));
}

void FdcnnNet::trainThreadImpl(const vector<InstancePtr> *instances,
    int_t thrd_id)
{
    LOG_INFO(("Start train thread with id = ")(thrd_id));
    int_t i1, i2;
    real_t z = 0;
    ulong_t rand_seed = static_cast<int_t>(thrd_id);
    int_t id = 0;
    real_t local_avgerr = 0;
    int_t local_iter = 0;
    while (true) {
        local_avgerr = 0;
        resetDiff(thrd_id);
        for (i1 = 0; i1 < batch_size; ++i1) {
            next_random(rand_seed);
            id = static_cast<int_t>(
                rand_seed % static_cast<ulong_t>(instances->size()));
            LookupTable(instances->at(id), sen_M[thrd_id]);
            lsen[thrd_id] = instances->at(id)->GetSenLen();
            for (i2 = 0; i2 < ncl; ++i2) {
                ndk[thrd_id][i2] = dynamicK(lsen[thrd_id], i2);
            }
            Forward(sen_M[thrd_id], lsen[thrd_id], conv_M[thrd_id],
                ncc[thrd_id], dropout_conv_mask_M[thrd_id], pool_V[thrd_id],
                ndk[thrd_id], pool_M[thrd_id], hid_V[thrd_id],
                dropout_hid_mask_V[thrd_id], out_V[thrd_id], rand_seed, true);
            //calculate error item for out_V
            for (i2 = 0, z = 0; i2 < nor; ++i2) {
                z += exp(out_V[thrd_id][i2]);
            }
            for (i2 = 0; i2 < nor; ++i2) {
                diff_out_V[thrd_id][i2] = exp(out_V[thrd_id][i2]) / z;
                local_avgerr += abs_s(diff_out_V[thrd_id][i2]);
            }
            int_t index = label_dict.find(instances->at(id)->label)->second;
            local_avgerr -= abs_s(diff_out_V[thrd_id][index]);
            diff_out_V[thrd_id][index] = diff_out_V[thrd_id][index] - 1;
            local_avgerr += abs_s(diff_out_V[thrd_id][index]);

            Backward(sen_M[thrd_id], diff_sen_M[thrd_id],
                lsen[thrd_id], conv_M[thrd_id], diff_conv_M[thrd_id],
                ncc[thrd_id], dropout_conv_mask_M[thrd_id], pool_V[thrd_id],
                ndk[thrd_id], pool_M[thrd_id], diff_pool_M[thrd_id],
                hid_V[thrd_id], diff_hid_V[thrd_id],
                dropout_hid_mask_V[thrd_id], diff_out_V[thrd_id],
                diff_MM[thrd_id], diff_WM[thrd_id], diff_U[thrd_id]);

            updateFeatureTable(diff_sen_M[thrd_id], instances->at(id));
        }
        updateParams(thrd_id);
        do {
            ScopedLock scoped_lock(cur_iter_mutex);
            local_iter = ++cur_iter;
            avgerr += local_avgerr;
        } while (false);
        if (0 == local_iter % LOG_PER_ITERS) {
            char buff[BUFF_SIZE];
            sprintf(buff, "thread id:%3d, iter_time:%6d, avgerr:%2.6f",
                thrd_id, local_iter, avgerr / LOG_PER_ITERS);
            LOG_INFO((buff));
            avgerr = 0;
        }
        if (0 == local_iter % snapshot) {
            ostringstream oss;
            oss << "snapshot_iter" << local_iter;
            bool ret = false;
            if (bin) { ret = SaveModelBinary(oss.str().c_str()); }
            else { ret = SaveModel(oss.str().c_str()); }
            if (ret) {
                LOG_INFO(("Save Snapshot succ, filename: ")(oss.str()));
            } else {
                LOG_ERROR(("Save Snapshot fail, filename: ")(oss.str()));
            }
        }
        if (local_iter >= iter_num) {
            break;
        }
    }
    LOG_INFO(("Finish train thread with id = ")(thrd_id));
}

void FdcnnNet::PredictTopK(InstancePtr instance, int_t k,
    std::vector<ProbType> &probs)
{
    LOG_DEBUG(("Enter PredictTopK"));
    if (k <= 0) {
        probs.clear();
        LOG_WARN(("K is less than 0."));
        return;
    }
    int_t thrd_id = registerRunRes();
    PredictTopKImpl(instance, k, probs, thrd_id);
    releaseRunRes(thrd_id);
    LOG_DEBUG(("Leave PredictTopK"));
}

void FdcnnNet::PredictTopKImpl(InstancePtr instance, int_t k,
    std::vector<ProbType> &probs, int_t thrd_id)
{
    LOG_DEBUG(("Enter PredictTopKImpl"));
    if (k <= 0) {
        probs.clear();
        LOG_WARN(("K is less than 0."));
        return;
    }
    if (k > nor) { k = nor; }
    static const pair<int_t, real_t> DEFAULT_FILL(-1, -DBL_MAX);
    int_t i1, i2;
    real_t z = 0;
    ulong_t rand_seed = 0;
    vector<pair<int_t, real_t> > index2prob(k, DEFAULT_FILL);
    LookupTable(instance, sen_M[thrd_id]);
    lsen[thrd_id] = instance->GetSenLen();
    for (i1 = 0; i1 < ncl; ++i1) {
        ndk[thrd_id][i1] = dynamicK(lsen[thrd_id], i1);
    }
    Forward(sen_M[thrd_id], lsen[thrd_id], conv_M[thrd_id], ncc[thrd_id], 0,
        pool_V[thrd_id], ndk[thrd_id], pool_M[thrd_id], hid_V[thrd_id], 0,
        out_V[thrd_id], rand_seed, false);
    for (i1 = 0, z = 0; i1 < nor; ++i1) {
        out_V[thrd_id][i1] = exp(out_V[thrd_id][i1]);
        z += out_V[thrd_id][i1];

        for (i2 = k - 2; i2 > -1; --i2) {
            if (out_V[thrd_id][i1] > index2prob[i2].second) {
                index2prob[i2 + 1] = index2prob[i2];
            }
            else { break; }
        }
        if (index2prob[i2 + 1].second < out_V[thrd_id][i1]) {
            index2prob[i2 + 1] = pair<int_t, real_t>(i1, out_V[thrd_id][i1]);
        }
    }
    for (i1 = 0; i1 < k; ++i1) {
        if (DEFAULT_FILL.first == index2prob[i1].first) { break; }
        index2prob[i1].second /= z;
    }
    k = i1;
    probs.resize(k);
    for (i1 = 0; i1 < k; ++i1) {
        probs[i1] = ProbType(index2label[index2prob[i1].first],
            index2prob[i1].second);
    }
    LOG_DEBUG(("Leave PredictTopKImpl"));
}

void FdcnnNet::GetFeatVec(InstancePtr instance, std::vector<real_t> &feat)
{
    LOG_DEBUG(("Enter GetFeatVec"));
    int_t thrd_id = registerRunRes();
    GetFeatVecImpl(instance, feat, thrd_id);
    releaseRunRes(thrd_id);
    LOG_DEBUG(("Leave GetFeatVec"));
}

void FdcnnNet::GetFeatVecImpl(InstancePtr instance, std::vector<real_t> &feat,
    int_t thrd_id)
{
    LOG_DEBUG(("Enter GetFeatVecImpl"));
    ulong_t rand_seed = 0;
    LookupTable(instance, sen_M[thrd_id]);
    lsen[thrd_id] = instance->GetSenLen();
    for (int_t i = 0; i < ncl; ++i) {
        ndk[thrd_id][i] = dynamicK(lsen[thrd_id], i);
    }
    Forward(sen_M[thrd_id], lsen[thrd_id], conv_M[thrd_id], ncc[thrd_id], 0,
        pool_V[thrd_id], ndk[thrd_id], pool_M[thrd_id], hid_V[thrd_id],
        0, out_V[thrd_id], rand_seed, false);
    feat.assign(hid_V[thrd_id][nhl - 1],
        hid_V[thrd_id][nhl - 1] + nhr[nhl - 1]);
    LOG_DEBUG(("Leave GetFeatVecImpl"));
}

void FdcnnNet::updateFeatureTable(real_t *diff_sen_M, InstancePtr instance)
{
    LOG_DEBUG(("Enter UpdateFeatureTable"));
    int_t i, j, k, l;
    int_t lsen = instance->GetSenLen();
    StrToIntMap::iterator it;
    for (i = 0; i < lsen; ++i) {
        for (j = 0, k = 0; j < nif; k += nfd[j], ++j) {
            if (j == 0) {
                if (isUsingWord2vec) {
                    LOG_DEBUG(("Ignore Word2vec feature"));
                    continue;
                }
            }
            it = feat_dicts[j].find(instance->features[j][i]);
            if (feat_dicts[j].end() == it) {
                it = feat_dicts[j].find(NIL);
            }
            for (l = 0; l < nfd[j]; ++l) {
                feature_table[j][it->second * nfd[j] + l] -= 
                    diff_sen_M[(l + k) * lsen + i] * alpha + 
                    feature_table[j][it->second * nfd[j] + l] * lambda;
                diff_sen_M[(l + k) * lsen + i] = 0;
            }
        }
    }
    LOG_DEBUG(("Leave UpdateFeatureTable"));
}

void* FdcnnNet::trainThread(void *param)
{
    ThreadFuncParam &func_param = *reinterpret_cast<ThreadFuncParam *>(param);
    func_param.net->trainThreadImpl(func_param.instances, func_param.thrd_id);
    return 0;
}

void FdcnnNet::Train(const vector<InstancePtr> &instances, int_t thrd_num,
    int_t batch_size, int_t iter_num, real_t alpha, real_t lambda,
    int_t snapshot)
{
    this->batch_size = batch_size;
    this->iter_num = iter_num;
    this->thrd_capacity = thrd_num;
    this->alpha = alpha;
    this->lambda = lambda;
    this->snapshot = snapshot;
    allocTrainRunMemory();
    vector<pthread_t> threads(thrd_num);
    vector<ThreadFuncParam> thread_params(thrd_num);
    int_t i = 0;

    for (i = 0; i < thrd_num; ++i){
        thread_params[i].net = this;
        thread_params[i].instances = &instances;
        thread_params[i].thrd_id = i;
        pthread_create(&threads[i], 0, &FdcnnNet::trainThread,
            reinterpret_cast<void *>(&thread_params[i]));
    }
    for (i = 0; i < thrd_num; ++i) {
        pthread_join(threads[i], 0);
    }
    freeRunMemory();
}

void FdcnnNet::makeIndex2label()
{
    LOG_DEBUG(("Enter MakeIndex2label"));
    index2label.resize(label_dict.size());
    for (StrToIntMap::iterator it = label_dict.begin(); label_dict.end() != it;
        ++it)
    {
        index2label[it->second] = it->first;
    }
    LOG_DEBUG(("Leave MakeIndex2label"));
}

void FdcnnNet::CreatePredictEnvironment(int_t thrd_capacity)
{
    LOG_DEBUG(("Enter CreatePredictEnvironment"));
    this->thrd_capacity = thrd_capacity;
    allocPredictRunMemory();
    LOG_DEBUG(("Leave CreatePredictEnvironment"));
}

}// namespace fdcnn

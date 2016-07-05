/*==============================================================================
 *   Copyright (C) 2016 All rights reserved.
 *
 *  File Name   : fdcnn_net.h
 *  Author      : Zhongping Liang
 *  Date        : 2016-07-05
 *  Version     : 1.0
 *  Description : This file provides declarations for FdcnnNet.
 *============================================================================*/

#ifndef FDCNN_NET_H_
#define FDCNN_NET_H_
#include <string>
#include <tr1/unordered_map>
#include <pthread.h>
#include <cstdlib>

#include "instance.h"
#include "operator.h"

namespace fdcnn
{

class FdcnnNet
{
public:
    typedef std::pair<std::string, real_t> ProbType;
    typedef std::tr1::unordered_map<std::string, uint_t> StrToIntMap;
    static const int_t  LOG_PER_ITERS;      // log per iterates.
    static const uint_t USLEEP_INTERVAL;    // usleep interval.

private :
    /*
     * struct ThreadFuncParam
     *  Helps thread function.
     */
    struct ThreadFuncParam {
        FdcnnNet* net;
        const std::vector<InstancePtr> *instances;
        int_t thrd_id;
    };

public:
    FdcnnNet();
    ~FdcnnNet();

protected :
    FdcnnNet(const FdcnnNet & instance);
    FdcnnNet & operator=(const FdcnnNet & instance);

public :
    /*
     *  @brief      Read configuration from config file.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      filname
     *  @return     true, if success; false, otherwise.
     */
    bool ReadConfig(const char *filename);

    /*
     *  @brief      Check configuration.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @return     true, if success; false, otherwise.
     */
    bool CheckConfig();

    /*
     *  @brief      Forword propagate.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      ...
     *  @return     void.
     */
    void Forward(real_t * sen_M, int_t lsen, real_t***conv_M, int_t * ncc,
        int_t ***dropout_conv_mask_M, int_t ***pool_V, int_t *ndk,
        real_t***pool_M, real_t ** hid_V, int_t **dropout_hid_mask_V,
        real_t *out_V, ulong_t &seed, bool isTrain);

    /*
     *  @brief      Backword propagate.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      ...
     *  @return     void.
     */
    void Backward(real_t * sen_M, real_t *diff_sen_M, int_t lsen,
        real_t ***conv_M, real_t ***diff_conv_M, int_t *ncc,
        int_t ***dropout_conv_mask_M, int_t ***pool_V, int_t *ndk,
        real_t*** pool_M, real_t*** diff_pool_M, real_t **hid_V,
        real_t**diff_hid_V, int_t **dropout_hid_mask_V, real_t * diff_out_V,
        real_t ***diff_MM, real_t **diff_WM, real_t *diff_U);

    /*
     *  @brief      Makedict from instances. Each feature has it's own buffer in
     *              range feature_table[i] the dictionary of each feature is
     *              feat_dicts[i] where i indicates the ith feature.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      instances : the input instances.
     *  @return     void.
     */
    void MakeDict(const std::vector<InstancePtr> &instances);

    /*
     *  @brief      Load word2vec from filename. Word feature is in the index of
     *              0, its dict is feat_dicts[0], and its feature value is saved
     *              in feature_table[0]. The function should be call after
     *              LoadModel or MakeDict.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      filename.
     *  @return     true, if success; false, otherwise.
     */
    bool LoadWord2vec(const char * filename);

    /*
     *  @brief      Save model into filename in text mode.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      filename.
     *  @return     true, if success; false, otherwise.
     */
    bool SaveModel(const char * filename);

    /*
     *  @brief      Load model from filename in text mode.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      filename.
     *  @return     true, if success; false, otherwise.
     */
    bool LoadModel(const char * filename);

    /*
     *  @brief      Save model into filename in binary mode.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      filename.
     *  @return     true, if success; false, otherwise.
     */
    bool SaveModelBinary(const char * filename);

    /*
     *  @brief      Load model from filename in binary mode.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      filename.
     *  @return     true, if success; false, otherwise.
     */
    bool LoadModelBinary(const char * filename);


    /*  Look up table: search dictionaries and store parameters in sen_M.
    *   Parameters :
    *       features: the iteration of features in dictionaries.
    *       sen_M    : the matrix of sentence
    *       lsen     : the length of sentence
    *   Return :
    *       void.
    */
    void LookupTable(InstancePtr instance, real_t *sen_M);

    /*
     *  @brief      Initialize the networks.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @return     void.
     */
    void Setup();

    /*
     *  @brief      Train model with instances.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      ...
     *  @return     void.
     */
    void Train(const std::vector<InstancePtr> &instances, int_t thrd_num,
        int_t batch_size, int_t iter_num, real_t alpha, real_t lambda,
        int_t snapshot);

    /*
     *  @brief      Create prediction environment.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      thrd_capacity: thread capacity.
     *  @return     void.
     */
    void CreatePredictEnvironment(int_t thrd_capacity);

    /*
     *  @brief      Predict for top k labels.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      instance: the input instance.
     *  @param      k: k most probability label.
     *  @param      probs: label and probability in descent order.
     *  @return     void.
     */
    void PredictTopK(InstancePtr instance, int_t k,
        std::vector<ProbType> &probs);
    void PredictTopKImpl(InstancePtr instance, int_t k,
        std::vector<ProbType> &probs, int_t thrd_id);

    /*
     *  @brief      Get feature, it's the top most hidden layer.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @param      instance: the input instance.
     *  @param      feat: the output feature.
     *  @return     void.
     */
    void GetFeatVec(InstancePtr instance, std::vector<real_t> &feat);
    void GetFeatVecImpl(InstancePtr instance, std::vector<real_t> &feat,
        int_t thrd_id);


    /*
     *  @brief      Get and Set functions.
     */
    inline void SetBinary(int_t binnary) { bin = binnary; }
    inline void SetUsingWord2vec(int_t yesOrNot) { isUsingWord2vec = yesOrNot; }
    inline int_t GetFldNum() { return nif; }
    inline int_t GetClassNum() { return nor; }
    inline int_t GetLabelId(const std::string &label) {
        StrToIntMap::iterator it = label_dict.find(label);
        if (label_dict.end() == it) { return -1; }
        return it->second;
    }
    inline void GetLabel(int_t id, std::string &label) {
        if (id < 0 || id > static_cast<int_t>(index2label.size())) {
            label.clear();
            return;
        }
        label.assign(index2label[id]);
    }

protected :
    inline real_t getRand(uint_t layer_size)
    {
        return (rand() / (real_t)RAND_MAX - 0.5) / sqrt(layer_size) * 2.0;
        //return (rand() / (real_t)RAND_MAX - 0.5) / layer_size;
    }

    int_t registerRunRes();
    void releaseRunRes(int_t id);
    void trainThreadImpl(const std::vector<InstancePtr> * instances,
        int_t thrd_id);
    inline int_t dynamicK(int_t lsen, int_t cncl)
    {
        return max_s(Ktop, static_cast<int_t>(
            lsen + ((real_t)(Ktop - lsen)) / ncl * (cncl + 1)));
    }
    void logInfoConfig();
    void updateParams(int_t thrd_id);
    void resetDiff(int_t thrd_id);
    void updateFeatureTable(real_t *sen_M, InstancePtr instance);
    void makeIndex2label();
    static void* trainThread(void *param);

    /*
     * @brief       Alloc and free memory.
     */
    void allocWeightMemory();
    void allocTrainRunMemory();
    void allocFeatureMemory();
    void allocPredictRunMemory();
    void freeFeatureMemory();
    void freeWeightMemory();
    void freeConfigMemory();
    void freeRunMemory();

protected:
    /* model parameters */
    //number of sentence dimensions
    int_t nsr;
    //number of convolution layers
    int_t ncl;
    //size of convolution window  per convolution layer, e.g. 11 8 5, with an
    //  attenuation of Watt
    int_t *scw;
    //number of convolution result rows per convolution layer, suggest calucated
    //  by max{Dtop, upper( (ncl - l) * nsr / ncl ) }
    int_t *ncr;
    //convolution filter matrix per convolution layer, there are ncm matrixs in
    //  each convolution layer, so the MM number is ncm[i] * ncm[i-1] , with
    //  dimension of
    //      MM[0]: dcr[0] * (scw[0] * nsr + 1)
    //      MM[i]: dcr[i] * (scw[i] * dcr[i-1] + 1)
    //  note a bias is added to it
    real_t ***MM;
    //number of conv_M matrix per convolution layer.
    int_t *ncm;
    //number of k-max pooling result in the top layer
    int_t Ktop;
    //number of hidden layers
    int_t nhl;
    //dimension of hidden result per layer
    int_t *nhr;
    //weight matrix per hidden layer WM[i],with dimension of
    //      WM[0] : nhr[0] * (nid + 1)
    //      WM[i] : nhr[i] * (nhr[i-1] + 1);
    //  note a bias is added to it.
    real_t **WM;
    //weight matrix of hidden to output layer, with dimension of
    //      U: nor * (nhr[nhl - 1] + 1);
    //  note a bias is added to it.
    real_t *U;
    //number of output layer dimension
    int_t nor;
    //number of each feature dimension
    int_t * nfd;
    //number of input features
    int_t nif;
    //featuer table.
    real_t **feature_table;
    //featuer dictionary.
    std::vector<StrToIntMap> feat_dicts;
    //label dict.
    StrToIntMap label_dict;
    //convert id to label.
    std::vector<std::string> index2label;

    /* run parameters see MM WM U above. */
    real_t **** diff_MM;
    real_t ***  diff_WM;
    real_t **   diff_U;

    /* Run memorys. */
    //sentence matrix, with dimention of nsr * MAX_SEN_LEN
    real_t ** sen_M;
    real_t ** diff_sen_M;
    //lenght of sentence
    int_t *lsen;
    //number of convolution result colmuns per convolution layer
    int_t ** ncc;
    //number of dynamic k-max pooling results per layer, calucated
    //  by max{Ktop, upper( (ncl - l) * lsen / ncl ) }
    int_t ** ndk;
    //convolution result matrix per convolution layer, each convolution layer
    //  have ncm matrixs, with dimension of 
    //      conv_M[0]: dcr[0] * (scw[0] + MAX_SEN_LEN -1)
    //      conv_M[i]: dcr[i] * (scw[i] +
    //          max{Ktop, upper( (ncl - i) * MAX_SEN_LEN / ncl ) } -1)
    real_t **** conv_M;
    real_t **** diff_conv_M;
    int_t  **** dropout_conv_mask_M;
    //k-max pooling result vector per convolution layer, each convolution layer
    //  have ncm results, stores the column number of conv_M
    int_t  **** pool_V;
    real_t **** pool_M;
    real_t **** diff_pool_M;
    //hidden layer vector per hidden layer, with dimension of
    //      hve[i] : nhl
    real_t *** hid_V;
    real_t *** diff_hid_V;
    int_t  *** dropout_hid_mask_V;
    //output vector of DCNN, with dimension of
    //      label2index.size()
    real_t ** out_V;      //output vector of DNN
    real_t ** diff_out_V; //output error term vector

    //dropout rate.
    real_t dropout_prob;
    //thread capacity.
    int_t thrd_capacity;

    /* parameters for train. */
    //batch size.
    int_t batch_size;
    //do snapshot when iter_num division snapshot.
    int_t snapshot;
    //learning rate
    real_t alpha;
    //regularization rate
    real_t lambda;
    //if using word2vec or not.
    int_t isUsingWord2vec;
    //average error.
    real_t avgerr;
    //current iterate number.
    int_t cur_iter;
    //mutex for cur_iter
    pthread_mutex_t cur_iter_mutex;

    /* other parameters. */
    //number of total iteration times for train.
    int_t iter_num;
    //save and load in binary mode
    int_t bin;

    /* To ensure thread safety. */
    std::vector<int_t> busy_map;
    int_t busy_num;
    pthread_mutex_t busy_num_mutex;

};

}// namespace fdcnn

#endif // FDCNN_NET_H_

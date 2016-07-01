/*******************************************************************
*  Copyright(c) 2014
*  All rights reserved.
*
*  File Name: sentence_model_base.h
*  Brief    : This file provides a sentence model basic class.
*  Current Version: 1.0
*  Author   : Zhongping Liang
*  Date     : 2014-12-12
******************************************************************/

#ifndef SENTENCE_MODELP_BASE_H
#define SENTENCE_MODELP_BASE_H
#include <string>
#include <tr1/unordered_map>
#include <pthread.h>

#include "instance.h"
#include "operator.h"

class FdcnnNet;

struct ThreadFuncParam {
    FdcnnNet* net;
    const std::vector<InstancePtr> *instances;
    int_t thrd_id;
};

class FdcnnNet
{
public:
    typedef std::pair<std::string, real_t> ProbType;

    typedef std::tr1::unordered_map<std::string, uint_t> StrToIntMap;

    static const int_t LOG_PER_ITERS;
public:
    FdcnnNet();
    ~FdcnnNet();

protected :
    FdcnnNet(const FdcnnNet & instance);
    FdcnnNet & operator=(const FdcnnNet & instance);

public :
    /*  Read configuration from config file.
    *   Parameters :
    *       void.
    *   Return :
    *       void.
    */
    bool ReadConfig(const char *filename);

    /*  Check configuration.
    *   Parameters :
    *       void.
    *   Return :
    *       void.
    */
    bool CheckConfig();

    /*  Forword propagate.
    *   Parameters :
    *       sen_M   : the matrix of input sentence.
    *       lsen    : the length of input sentence.
    *       conv_M  : the matrix of convolution result .
    *       ncc     : the number of convolution result colmuns per convolution
    *                 layer.
    *       kpl_V   : the vector of k-max pooling result.
    *       ndk     : the number of dynamic k-max pooling results per layer
    *       hid_V   : the vector of hidden layer.
    *       out_V   : the vector of out layer.
    *   Return :
    *       void.
    */
    void Forward(real_t * sen_M, int_t lsen, real_t***conv_M, int_t * ncc,
        int_t ***dropout_conv_mask_M, int_t ***pool_V, int_t *ndk,
        real_t***pool_M, real_t ** hid_V, int_t **dropout_hid_mask_V,
        real_t *out_V, ulong_t &seed, bool isTrain);
    /*  Backward propagate.
    *   Parameters :
    *       sen_M    : the matrix of input sentence.
    *       diff_sen_M     : the matrix of input sentence's error term.
    *       lsen     : the length of input sentence.
    *       conv_M      : the matrix of convolution result.
    *       diff_conv_M     : the matrix of convolution result's error term.
    *       ncc     : the number of convolution result colmuns per convolution 
    *                 layer.
    *       kpl_V     : the vector of k-max pooling result.
    *       ndk     : the number of dynamic k-max pooling results per layer
    *       hid_V     : the vector of hidden layer.
    *       diff_hid_V     : the vector of hidden layer's error term.
    *       loz     : the vector of out layer's error term.
    *   Return :
    *       void.
    */
    void Backward(real_t * sen_M, real_t *diff_sen_M, int_t lsen,
        real_t ***conv_M, real_t ***diff_conv_M, int_t *ncc,
        int_t ***dropout_conv_mask_M, int_t ***pool_V, int_t *ndk,
        real_t*** pool_M, real_t*** diff_pool_M, real_t **hid_V,
        real_t**diff_hid_V, int_t **dropout_hid_mask_V, real_t * diff_out_V,
        real_t ***diff_MM, real_t **diff_WM, real_t *diff_U);

    /*  Make dictionary : each feature is has it's own buffer in range 
    *   feature_table[0, nif) the dictionary of each feature is dicts[i]
    *   where i indicates the ith feature.
    *   Parameters :
    *       void.
    *   Return :
    *       void.
    */
    void MakeDict(const std::vector<InstancePtr> &instances);

    /*  Make dictionary : each feature is has it's own buffer in range 
    *   feature_table[0, nif) the dictionary of each feature is dicts[i] where 
    *   i indicates the ith feature, the word2vec dict is in dicts[0].
    *   Parameters :
    *       void.
    *   Return :
    *       void.
    */
    bool LoadWord2vec(const char * filename);

    /*  Save model in model_file_name.
    *   Parameters :
    *       void.
    *   Return :
    *       void.
    */
    virtual bool SaveModel(const char * filename);

    /*  Load model from model_file_name.
    *   Parameters :
    *       void.
    *   Return :
    *       void.
    */
    virtual bool LoadModel(const char * filename);

    /*  Save model in model_file_name.
    *   Parameters :
    *       void.
    *   Return :
    *       void.
    */
    virtual bool SaveModelBinary(const char * filename);

    /*  Load model from model_file_name.
    *   Parameters :
    *       void.
    *   Return :
    *       void.
    */
    virtual bool LoadModelBinary(const char * filename);


    /*  Look up table: search dictionaries and store parameters in sen_M.
    *   Parameters :
    *       features: the iteration of features in dictionaries.
    *       sen_M    : the matrix of sentence
    *       lsen     : the length of sentence
    *   Return :
    *       void.
    */
    void LookupTable(InstancePtr instance, real_t *sen_M);

    void Train(const std::vector<InstancePtr> &instances, int_t thrd_num,
        int_t batch_size, int_t iter_num, real_t alpha, real_t lambda,
        int_t snapshot);
    

    void PredictTopK(InstancePtr instance, int_t k,
        std::vector<ProbType> &probs);
    void GetFeatVec(InstancePtr instance, std::vector<real_t> &feat);


    void PredictTopKImpl(InstancePtr instance, int_t k,
        std::vector<ProbType> &probs, int_t thrd_id);
    void GetFeatVecImpl(InstancePtr instance, std::vector<real_t> &feat,
        int_t thrd_id);


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

    void CreatePredictEnvironment(int_t thrd_capacity);

    /*  Initialize the networks.
    *   Parameters :
    *       void.
    *   Return :
    *       void.
    */
    void Setup();
protected :
    /*  Get a float random value: the return value is in 
     *  range(-1/layer_size, 1/layer_size).
     *  Parameters :
     *      layer_size  : Limit the range of random value.
     *  Return :
     *      float :a float random value in range(-1/layer_size, 1/layer_size).
     */
    inline real_t getRand(uint_t layer_size)
    {
        return (rand() / (real_t)RAND_MAX - 0.5) / sqrt(layer_size) * 2.0;
        //return (rand() / (real_t)RAND_MAX - 0.5) / layer_size;
    }

    int_t registerRunRes();
    void releaseRunRes(int_t id);
    void trainThreadImpl(const std::vector<InstancePtr> * instances,
        int_t thrd_id);




    /*  Allocate memory for parameters of networks.
    *   Parameters :
    *       void.
    *   Return :
    *       void.
    */
    void allocWeightMemory();

    void allocFeatureMemory();
    void freeFeatureMemory();

    void allocTrainRunMemory();
    /*  Release memory for parameters of networks and dictionaries.
    *   Parameters :
    *       void.
    *   Return :
    *       void.
    */
    //void freeParamMemory();
    void freeWeightMemory();
    void freeConfigMemory();
    void freeRunMemory();

    void allocPredictRunMemory();

    /*  Calculate pooling result number.
    *   Parameters :
    *       lsen : the length of sentence.
    *       cncl    : current convolution layer number
    *   Return :
    *       void.
    */
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
    static const uint_t USLEEP_INTERVAL;
protected:
    //model parameters
    int_t nsr;            //number of sentence dimensions
    int_t ncl;            //number of convolution layers
    int_t *scw;           //size of convolution window  per convolution layer,11 8 5, with a attenuation of Watt
    int_t *ncr;           //number of convolution result rows per convolution layer, maybe calucateed by max{Dtop, upper( (ncl - l) * nwd / ncl ) } 
    real_t ***MM;         //convolution filter matrix per convolution layer, there are ncm matrixs in each convolution layer, so the MM number is ncm[i] * ncm[i-1] , with dimension of -MM[0]:dcr[0] * (scw[0] * nwd + 1) -MM[i]:dcr[i] * (scw[i] * dcr[i-1] + 1) ; note a bias is added to it
    int_t *ncm;           //number of conv_M matrix per convolution layer.
    int_t Ktop;           //number of k-max pooling result in the top layer
    int_t nhl;            //number of hidden layers
    int_t *nhr;           //dimension of hidden result per layer
    real_t **WM;              //weight matrix per hidden layer WM[i],with dimension of -WM[0] : nhr[0] * (nid + 1) -WM[i] : nhr[i] * (nhr[i-1] + 1); note a bias is added to it
    real_t *U;                //weight matrix of hidden to output layer, with dimension of U: nor * (nhr[nhl - 1] + 1);  note a bias is added to it
    int_t nor;            //number of output layer dimension
    int_t * nfd;          //number of each feature dimension
    int_t nif;            //number of input features
    real_t **feature_table;   //featuer table.

    //run parameters
    real_t ****diff_MM;
    real_t ***diff_WM;
    real_t **diff_U;

    real_t ** sen_M;            //sentence matrix, with dimention of nsr * MAX_SEN_LEN
    real_t ** diff_sen_M;   //error of sentence matrix
    int_t *lsen;            //lenght of sentence
    int_t ** ncc;          //number of convolution result colmuns per convolution layer 
    int_t ** ndk;          //number of dynamic k-max pooling results per layer, calucateed by max{Ktop, upper( (ncl - l) * lsen / ncl ) } 
    real_t **** conv_M;        //convolution result matrix per convolution layer, each convolution layer have ncm matrixs, with dimension of -conv_M[0]: dcr[0] * (scw[0] + MAX_SEN_LEN -1) -conv_M[i]:  dcr[i] * (scw[i] + max{Ktop, upper( (ncl - i) * MAX_SEN_LEN / ncl ) } -1)
    real_t **** diff_conv_M;       //convolution error term matrix per convolution layer, each convolution layer have ncm matrixs, with dimension of -conv_M[0]: dcr[0] * (scw[0] + MAX_SEN_LEN -1) -conv_M[i]:  dcr[i] * (scw[i] + max{Ktop, upper( (ncl - i) * MAX_SEN_LEN / ncl ) } -1)
    int_t ****dropout_conv_mask_M;
    int_t **** pool_V;        //k-max pooling result vector per convolution layer, each convolution layer have ncm results, stores the column number of conv_M
    real_t **** pool_M;
    real_t **** diff_pool_M;
    real_t *** hid_V;        //hidden layer vector per hidden layer,with dimension of  hve[i]:nhl
    real_t *** diff_hid_V;        //hidden laryer error term vector per hidden layer,with dimension of  hve[i]:nhl
    int_t ***dropout_hid_mask_V;
    real_t ** out_V;         //output vector of DNN
    real_t ** diff_out_V;         //output error term vector
    real_t dropout_prob;
    int_t thrd_capacity;
    //parameters for train
    int_t batch_size;
    int_t snapshot;
    real_t alpha;               //learning rate
    real_t lambda;               //regularization rate

    int_t isUsingWord2vec;

    //other parameters
    int_t iter_num;         //number of iteration times
    //int_t thrd_num;         //number of threads.
    std::vector<StrToIntMap> feat_dicts;
    StrToIntMap label_dict;
    std::vector<std::string> index2label;
    int_t bin;              // binary save or load mode

    std::vector<int_t> busy_map;

    int_t busy_num;
    pthread_mutex_t busy_num_mutex;
    real_t avgerr;
    int_t cur_iter;
    pthread_mutex_t cur_iter_mutex;
};
#endif


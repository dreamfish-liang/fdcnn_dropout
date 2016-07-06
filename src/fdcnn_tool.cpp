/*==============================================================================
 *   Copyright (C) 2016 All rights reserved.
 *
 *  File Name   : fdcnn_tool.cpp
 *  Author      : Zhongping Liang
 *  Date        : 2016-07-05
 *  Version     : 1.0
 *  Description : This file provides command line tools for fdcnn.
 *============================================================================*/
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <cstring>

#include "string_handler.h"
#include "fdcnn_net.h"
#include "instance.h"

using namespace std;
using namespace fdcnn;

const int_t MAX_PATH_LEN = 100;

bool LoadInstances(const char *filename, int_t nflds,
    std::vector<InstancePtr> &instances)
{
    ifstream ifs(filename, fstream::in | fstream::binary);
    if (!ifs.good())
    {
        LOG_ERROR(("Open Instance file failed, filename :")(filename));
        return false;
    }
    LOG_INFO(("Start Loading instances."));
    ScopedFstream<ifstream> scoped_ifs(ifs);  //auto close
    string line;
    vector<string> flds;
    uint_t linu = 0;
    InstancePtr instance(new Instance(nflds));
    do
    {
        getline(ifs, line);
        ++linu;
        StringHandler::RightTrimString(line, line, "\n\r");
        if (line.empty())
        {
            if (!ifs.good()) { break; }
            else             { continue; }
        }
        StringHandler::SplitString(line, flds, "\t");
        if (1 == flds.size())
        {
            instance->label.assign(flds[0]);
            if (instance->GetSenLen() > MAX_SEN_LEN) {
                LOG_ERROR(("Instance too long."));
            }
            else {
                instances.push_back(instance);
            }
            instance.reset(new Instance(nflds));
        }
        else if (1 + nflds == static_cast<int_t>(flds.size()))
        {
            for (uint_t i = 1; i < flds.size(); ++i)
            {
                instance->features[i - 1].push_back(flds[i]);
            }
        }
        else
        {
            LOG_ERROR(("Field number not right in file: ")(filename)
                (", line number is: ")(linu)
                (", content is: ")(line));
            return false;
        }
    } while (ifs.good());
    LOG_INFO(("Finish Loading instances."));
    return true;
}

bool SavePredResult(const char* filename, vector<InstancePtr> &instances,
    vector<vector<FdcnnNet::ProbType> > &probs)
{
    if (instances.empty()) { return false; }
    ofstream ofs(filename, fstream::out | fstream::binary);
    if (!ofs.good()) {
        LOG_ERROR(("Open PredResult file failed, filename: ")(filename));
        return false;
    }
    LOG_INFO(("Start Saving Predict result."));
    ScopedFstream<ofstream> scoped_ofs(ofs);
    string line;
    int_t i1, i2, i3, lsen, nflds, size1, size2;
    nflds = instances.front()->GetFldNum();
    size1 = static_cast<int_t>(instances.size());
    for (i1 = 0; i1 < size1; ++i1) {
        lsen = instances[i1]->GetSenLen();
        for (i2 = 0; i2 < lsen; ++i2) {
            for (i3 = 0; i3 < nflds; ++i3) {
                if (i3) { ofs << '\t'; }
                ofs << instances[i1]->features[i3][i2];
            }
            ofs << '\n';
        }
        ofs << instances[i1]->label;
        size2 = static_cast<int_t>(probs[i1].size());
        for (i2 = 0; i2 < size2; ++i2) {
            ofs << '\t' << probs[i1][i2].first << ':' << probs[i1][i2].second;
        }
        ofs << '\n' << endl;
    }
    LOG_INFO(("Finish Saving Predict result."));
    return true;
}

int_t GetArgPos(char *str, int_t argc, char **argv) {
    int_t a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            fprintf(stdout, "Argument missing for %s\n", str);
            exit(-1);
        }
        return a;
    }
    return -1;
}

void Train(int_t argc, char **argv){
    if (argc < 3) {
        fprintf(stdout, "Fdcnn toolkit v 0.1b\n\n");
        fprintf(stdout, "Train options:\n");
        fprintf(stdout, "Parameters for Train:\n");
        fprintf(stdout, "\t-train <file>\n");
        fprintf(stdout, "\t\tUse text data from <file> to train the model\n");
        fprintf(stdout, "\t-config <file>\n");
        fprintf(stdout, "\t\tUse configuration from <file> to construct the"
            " model\n");
        fprintf(stdout, "\t-model <file>\n");
        fprintf(stdout, "\t\tUse <file> to save the model\n");
        fprintf(stdout, "\t-recover <file>\n");
        fprintf(stdout, "\t\tUse <file> to recover the model\n");
        fprintf(stdout, "\t-word2vec <file>\n");
        fprintf(stdout, "\t\tUse <file> to load the word2vec\n");
        fprintf(stdout, "\t-iter <int_t>\n");
        fprintf(stdout, "\t\tUse <int_t> iteration number (default 10000)\n");
        fprintf(stdout, "\t-batch_size <int_t>\n");
        fprintf(stdout, "\t\tUse <int_t> batch_size (default 64)\n");
        fprintf(stdout, "\t-threads <int_t>\n");
        fprintf(stdout, "\t\tUse <int_t> threads (default 1)\n");
        fprintf(stdout, "\t-alpha <float>\n");
        fprintf(stdout, "\t\tSet the learning rate <float>; (default is 0.01)"
            "\n");
        fprintf(stdout, "\t-lambda <float>\n");
        fprintf(stdout, "\t\tSet the regularization rate <float>; "
            "(default is 0, suguesting not larger than 1e-6)\n");
        fprintf(stdout, "\t-binary <int_t>\n");
        fprintf(stdout, "\t\tSet save in binary mode; (default is 1)\n");
        fprintf(stdout, "\t-snapshot <int_t>\n");
        fprintf(stdout, "\t\tSave model every <int_t> iter; (default 5000)\n");
        fprintf(stdout, "For example:\n");
        fprintf(stdout, "\t./fdcnn_train -train train.tsv -config"
            " config.txt -model model -iter 1000 -alpha 0.01 -lambda 0"
            " -binary 0 -threads 1\n");
        exit(-1);
    }
    int_t i;
    char train_file_name[MAX_PATH_LEN] = { 0 };
    char config_file_name[MAX_PATH_LEN] = { 0 };
    char model_file_name[MAX_PATH_LEN] = { 0 };
    char recover_file_name[MAX_PATH_LEN] = { 0 };
    char word2vec_file_name[MAX_PATH_LEN] = { 0 };
    int_t iter_num = 10000;
    int_t thrd_num = 1;
    int_t batch_size = 64;
    real_t alpha = 0.01;
    real_t lambda = 0;
    int_t binary = 1;
    int_t snapshot = 5000;
    vector<InstancePtr> instances;
    if ((i = GetArgPos((char *)"-train", argc, argv)) > 0) {
        strcpy(train_file_name, argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-config", argc, argv)) > 0) {
        strcpy(config_file_name, argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-model", argc, argv)) > 0) {
        strcpy(model_file_name, argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-recover", argc, argv)) > 0) {
        strcpy(recover_file_name, argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-word2vec", argc, argv)) > 0) {
        strcpy(word2vec_file_name, argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-iter", argc, argv)) > 0) {
        iter_num = atoi(argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-batch_size", argc, argv)) > 0) {
        batch_size = atoi(argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-threads", argc, argv)) > 0) {
        thrd_num = atoi(argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-alpha", argc, argv)) > 0) {
        alpha = atof(argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-lambda", argc, argv)) > 0) {
        lambda = atof(argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-binary", argc, argv)) > 0) {
        binary = atoi(argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-snapshot", argc, argv)) > 0) {
        snapshot = atoi(argv[i + 1]);
    }
    if (train_file_name[0] == 0){
        fprintf(stderr, "Train file must be given in the paramters\n");
        exit(-1);
    }
    if (config_file_name[0] == 0){
        fprintf(stderr, "Config file must be given in the paramters\n");
        exit(-1);
    }
    if (model_file_name[0] == 0){
        fprintf(stderr, "Model file must be given in the paramters\n");
        exit(-1);
    }
    fprintf(stdout, "Parameters:\n");
    fprintf(stdout, "\tTrain file name = %s\n", train_file_name);
    fprintf(stdout, "\tConfiguration file name = %s\n", config_file_name);
    fprintf(stdout, "\tModel file name = %s\n", model_file_name);
    fprintf(stdout, "\tRecover file name = %s\n", recover_file_name);
    fprintf(stdout, "\tWord2vec file name = %s\n", word2vec_file_name);
    fprintf(stdout, "\tIteration number = %d\n", iter_num);
    fprintf(stdout, "\tBatch size = %d\n", batch_size);
    fprintf(stdout, "\tBinary = %d\n", binary);
    fprintf(stdout, "\tThread number = %d\n", thrd_num);
    fprintf(stdout, "\tLearning rate = %lf\n", alpha);
    fprintf(stdout, "\tRegularization rate = %lf\n", lambda);
    fprintf(stdout, "\tsnapshot = %d\n", snapshot);
    FdcnnNet net;
    bool ret = false;
    ret = net.ReadConfig(config_file_name);
    if (!ret){
        fprintf(stderr, "Read config file failed!\n");
        exit(-1);
    }
    ret = net.CheckConfig();
    if (!ret){
        fprintf(stderr, "Check config failed!\n");
        exit(-1);
    }
    ret = LoadInstances(train_file_name, net.GetFldNum(), instances);
    if (!ret){
        fprintf(stderr, "Load instances failed!\n");
        exit(-1);
    }
    if (recover_file_name[0]) {
        if (binary) { ret = net.LoadModelBinary(recover_file_name); }
        else { ret = net.LoadModel(recover_file_name); }
        if (!ret){
            fprintf(stderr, "Recover model failed!\n");
            exit(-1);
        }
    } else {
        net.MakeDict(instances);
        net.Setup();
    }
    if (word2vec_file_name[0]) {
        ret = net.LoadWord2vec(word2vec_file_name);
        if (!ret){
            fprintf(stderr, "Load word2vec Failed!\n");
            exit(-1);
        }
    }
    net.SetBinary(binary);
    net.Train(instances, thrd_num, batch_size, iter_num, alpha, lambda,
        snapshot);
    if (binary) { ret = net.SaveModelBinary(model_file_name); }
    else { ret = net.SaveModel(model_file_name); }
    if (!ret){
        fprintf(stderr, "Save model Failed!\n");
        exit(-1);
    }
}

struct PredictThreadParam
{
    FdcnnNet *net;
    vector<InstancePtr> *instances;
    vector<vector<FdcnnNet::ProbType> > *probs;
    int_t * cur_inst;
    pthread_mutex_t *cur_inst_mutex;
    int_t thrd_id;
};

void predictThreadImpl(PredictThreadParam &param)
{
    int_t inst;
    const int_t size = static_cast<int_t>(param.instances->size());
    while (true) {
        do {
            ScopedLock scoped_lock(*(param.cur_inst_mutex));
            if (*(param.cur_inst) >= size) {
                return;
            }
            inst = (*(param.cur_inst))++;
        } while (false);
        param.net->PredictTopKImpl(param.instances->at(inst),
            param.net->GetClassNum(), param.probs->at(inst), param.thrd_id);
    }
}


void* predictThread(void *param)
{
    PredictThreadParam &func_param =
        *reinterpret_cast<PredictThreadParam*>(param);
    LOG_INFO(("Start predict thread with id = ")(func_param.thrd_id));
    predictThreadImpl(func_param);
    LOG_INFO(("Finish predict thread with id = ")(func_param.thrd_id));
    return 0;
}

void Predict(int_t argc, char **argv){
    if (argc < 3) {
        fprintf(stdout, "Fdcnn toolkit v 0.1b\n\n");
        fprintf(stdout, "Predict options:\n");
        fprintf(stdout, "Parameters for Predict:\n");
        fprintf(stdout, "\t-predict <file>\n");
        fprintf(stdout, "\t\tUse text data from <file> to train the model\n");
        fprintf(stdout, "\t-model <file>\n");
        fprintf(stdout, "\t\tUse <file> to save the model\n");
        fprintf(stdout, "\t-out <file>\n");
        fprintf(stdout, "\t\tUse <file> to save the predict result\n");
        fprintf(stdout, "\t-threads <int_t>\n");
        fprintf(stdout, "\t\tUse <int_t> threads (default 1)\n");
        fprintf(stdout, "\t-binary <int_t>\n");
        fprintf(stdout, "\t\tSet save in binary mode; (default is 1)\n");
        fprintf(stdout, "For example:\n");
        fprintf(stdout, "\t./fdcnn_predict -predict test.tsv -model"
            " model -out out.tsv -binary 0 -threads 1\n");
        exit(-1);
    }
    int_t i;
    char predict_in_file_name[MAX_PATH_LEN] = { 0 };
    char model_file_name[MAX_PATH_LEN] = { 0 };
    char predict_out_file_name[MAX_PATH_LEN] = { 0 };
    int_t thrd_num = 1;
    int_t binary = 1;
    if ((i = GetArgPos((char *)"-predict", argc, argv)) > 0) {
        strcpy(predict_in_file_name, argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-model", argc, argv)) > 0) {
        strcpy(model_file_name, argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-out", argc, argv)) > 0) {
        strcpy(predict_out_file_name, argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-threads", argc, argv)) > 0) {
        thrd_num = atoi(argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-binary", argc, argv)) > 0) {
        binary = atoi(argv[i + 1]);
    }

    if (predict_in_file_name[0] == 0){
        fprintf(stderr, "Predict file must be given in the paramters\n");
        exit(-1);
    }
    if (model_file_name[0] == 0){
        fprintf(stderr, "Model file must be given in the paramters\n");
        exit(-1);
    }

    fprintf(stdout, "Parameters:\n");
    fprintf(stdout, "\tPredict in file name = %s\n", predict_in_file_name);
    fprintf(stdout, "\tModel file name = %s\n", model_file_name);
    fprintf(stdout, "\tPredict out file name = %s\n", predict_out_file_name);
    fprintf(stdout, "\tThread number = %d\n", thrd_num);
    fprintf(stdout, "\tBinary = %d\n", binary);
    FdcnnNet net;
    net.SetBinary(binary);
    bool ret = false;
    if (binary) { ret = net.LoadModelBinary(model_file_name); } 
    else { ret = net.LoadModel(model_file_name); }
    if (!ret) {
        fprintf(stderr, "Load Model Failed.\n");
        exit(-1);
    }
    net.CreatePredictEnvironment(thrd_num);

    vector<InstancePtr> instances;
    LoadInstances(predict_in_file_name, net.GetFldNum(), instances);

    int_t cur_inst = 0;
    pthread_mutex_t cur_inst_mutex;
    pthread_mutex_init(&cur_inst_mutex, NULL);
    vector<vector<FdcnnNet::ProbType> > probs(instances.size());

    vector<pthread_t> threads(thrd_num);
    vector<PredictThreadParam> thread_params(thrd_num);
    for (i = 0; i < thrd_num; ++i){
        thread_params[i].net = &net;
        thread_params[i].instances = &instances;
        thread_params[i].probs = &probs;
        thread_params[i].cur_inst = &cur_inst;
        thread_params[i].cur_inst_mutex = &cur_inst_mutex;
        thread_params[i].thrd_id = i;
        pthread_create(&threads[i], 0, &predictThread,
            reinterpret_cast<void *>(&thread_params[i]));
    }
    for (i = 0; i < thrd_num; ++i) {
        pthread_join(threads[i], 0);
    }

    int_t class_num = net.GetClassNum();
    vector<int_t> hits_vec(class_num, 0);
    vector<int_t> pred_vec(class_num, 0);
    vector<int_t> gold_vec(class_num, 0);
    int_t total = 0;
    int_t gold_index = 0;
    int_t pred_index = 0;
    int_t hit_num = 0;
    real_t p, r, f;

    int_t size = static_cast<int_t>(instances.size());
    for (i = 0; i < size; ++i) {
        gold_index = net.GetLabelId(instances[i]->label);
        pred_index = net.GetLabelId(probs[i].front().first);
        if (-1 != gold_index) { ++gold_vec[gold_index]; }
        if (gold_index == pred_index) { ++hits_vec[pred_index]; }
        ++pred_vec[pred_index];
    }
    string label;
    fprintf(stdout, "\n"
        "======================================================================"
        "==================\n");
    for (i = 0; i < class_num; ++i) {
        net.GetLabel(i, label);
        hit_num += hits_vec[i];
        p = !pred_vec[i] ? 0 : (real_t)hits_vec[i] / (real_t)pred_vec[i];
        r = !gold_vec[i] ? 0 : (real_t)hits_vec[i] / (real_t)gold_vec[i];
        f = p + r == 0 ? 0.0 : 2.0 * p * r / (p + r);
        fprintf(stdout, "Label %s: pred = %-6d, gold = %-6d, hits = %-6d,"
            " pre = %2.6lf, rec = %-2.6lf, f1 = %-2.6lf\n"
            , label.c_str(), pred_vec[i], gold_vec[i], hits_vec[i]
            , p, r, f);
    }
    total = instances.size();
    fprintf(stdout, "Total : all = %-6d, right = %-6d, acc = %-2.6lf\n", 
        total, hit_num, !total ? 0 : (real_t)hit_num / total);

    fprintf(stdout, "\n"
        "======================================================================"
        "==================\n");
    if (predict_out_file_name[0]) {
        ret = SavePredResult(predict_out_file_name, instances, probs);
        if (!ret) {
            fprintf(stderr, "Save predict result falied.\n");
        }
    }
}

struct GetFeatThreadParam
{
    FdcnnNet *net;
    vector<InstancePtr> *instances;
    vector<vector<real_t> > *feats;
    int_t * cur_inst;
    pthread_mutex_t *cur_inst_mutex;
    int_t thrd_id;
};

void getFeatThreadImpl(GetFeatThreadParam &param)
{
    int_t inst;
    const int_t size = static_cast<int_t>(param.instances->size());
    while (true) {
        do {
            ScopedLock scoped_lock(*(param.cur_inst_mutex));
            if (*(param.cur_inst) >= size) {
                return;
            }
            inst = (*(param.cur_inst))++;
        } while (false);
        param.net->GetFeatVecImpl(param.instances->at(inst),
            param.feats->at(inst), param.thrd_id);
    }
}

void* getFeatThread(void *param)
{
    GetFeatThreadParam &func_param =
        *reinterpret_cast<GetFeatThreadParam*>(param);
    LOG_INFO(("Start getfeat thread with id = ")(func_param.thrd_id));
    getFeatThreadImpl(func_param);
    LOG_INFO(("Finish getfeat thread with id = ")(func_param.thrd_id));
    return 0;
}

bool SaveFeatResult(const char *filename, vector<vector<real_t> > &feats){
    if (feats.empty()) { return false; }
    ofstream ofs(filename, fstream::out | fstream::binary);
    if (!ofs.good()) {
        LOG_ERROR(("Open FeatResult file failed, filename: ")(filename));
        return false;
    }
    LOG_INFO(("Start Saving Feat result."));
    ScopedFstream<ofstream> scoped_ofs(ofs);
    for (vector<vector<real_t> >::size_type i = 0; i < feats.size(); ++i) {
        for (vector<real_t>::size_type j = 0; j < feats[i].size(); ++j) {
            if (j) { ofs << '\t'; }
            ofs << feats[i][j];
        }
        ofs << endl;
    }
    LOG_INFO(("Finish Saving Feat result."));
    return true;
}

void SwitchModel(int_t argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        fprintf(stdout, "Fdcnn toolkit v 0.1b\n\n");
        fprintf(stdout, "Switch model options:\n");
        fprintf(stdout, "For example:\n");
        fprintf(stdout, "\t./fdcnn_switch_model model.bin model.txt\n");
        fprintf(stdout, "\t./fdcnn_switch_model some model.txt model.bin\n");
        exit(-1);
    }
    char src_file_name[MAX_PATH_LEN] = { 0 };
    char dst_file_name[MAX_PATH_LEN] = { 0 };
    strcpy(src_file_name, argv[argc - 2]);
    strcpy(dst_file_name, argv[argc - 1]);
    FdcnnNet net;
    bool ret = false;
    if (3 == argc) {
        ret = net.LoadModelBinary(src_file_name);
        if (!ret) {
            fprintf(stderr, "Load model failed.\n");
            exit(-1);
        }
        ret = net.SaveModel(dst_file_name);
        if (!ret) {
            fprintf(stderr, "Save model failed.\n");
            exit(-1);
        }
    }
    else if (4 == argc) {
        ret = net.LoadModel(src_file_name);
        if (!ret) {
            fprintf(stderr, "Load model failed.\n");
            exit(-1);
        }
        ret = net.SaveModelBinary(dst_file_name);
        if (!ret) {
            fprintf(stderr, "Save model failed.\n");
            exit(-1);
        }
    }
}

void GetFeat(int_t argc, char **argv){
    if (argc < 3) {
        fprintf(stdout, "Fdcnn toolkit v 0.1b\n\n");
        fprintf(stdout, "GetFeats options:\n");
        fprintf(stdout, "Parameters for GetFeat:\n");
        fprintf(stdout, "\t-predict <file>\n");
        fprintf(stdout, "\t\tUse text data from <file> to train the model\n");
        fprintf(stdout, "\t-model <file>\n");
        fprintf(stdout, "\t\tUse <file> to save the model\n");
        fprintf(stdout, "\t-out <file>\n");
        fprintf(stdout, "\t\tUse <file> to save the predict result\n");
        fprintf(stdout, "\t-threads <int_t>\n");
        fprintf(stdout, "\t\tUse <int_t> threads (default 1)\n");
        fprintf(stdout, "\t-binary <int_t>\n");
        fprintf(stdout, "\t\tSet save in binary mode; (default is 1)\n");
        fprintf(stdout, "For example:\n");
        fprintf(stdout, "\t./fdcnn_getfeat -predict test.tsv -model"
            " model -out out.fea -binary 0 -threads 1\n");
        exit(-1);
    }
    int_t i;
    char predict_in_file_name[MAX_PATH_LEN] = { 0 };
    char model_file_name[MAX_PATH_LEN] = { 0 };
    char feat_out_file_name[MAX_PATH_LEN] = { 0 };
    int_t thrd_num = 1;
    int_t binary = 1;
    if ((i = GetArgPos((char *)"-predict", argc, argv)) > 0) {
        strcpy(predict_in_file_name, argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-model", argc, argv)) > 0) {
        strcpy(model_file_name, argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-out", argc, argv)) > 0) {
        strcpy(feat_out_file_name, argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-threads", argc, argv)) > 0) {
        thrd_num = atoi(argv[i + 1]);
    }
    if ((i = GetArgPos((char *)"-binary", argc, argv)) > 0) {
        binary = atoi(argv[i + 1]);
    }

    if (predict_in_file_name[0] == 0){
        fprintf(stderr, "Predict file must be given in the paramters\n");
        exit(-1);
    }
    if (model_file_name[0] == 0){
        fprintf(stderr, "Model file must be given in the paramters\n");
        exit(-1);
    }
    if (feat_out_file_name[0] == 0){
        fprintf(stderr, "Model file must be given in the paramters\n");
        exit(-1);
    }

    fprintf(stdout, "Parameters:\n");
    fprintf(stdout, "\tPredict in file name = %s\n", predict_in_file_name);
    fprintf(stdout, "\tModel file name = %s\n", model_file_name);
    fprintf(stdout, "\tFeature out file name = %s\n", feat_out_file_name);
    fprintf(stdout, "\tThread number = %d\n", thrd_num);
    fprintf(stdout, "\tBinary = %d\n", binary);
    FdcnnNet net;
    net.SetBinary(binary);
    bool ret = false;
    if (binary) { ret = net.LoadModelBinary(model_file_name); }
    else { ret = net.LoadModel(model_file_name); }
    if (!ret) {
        fprintf(stderr, "Load Model Failed.\n");
        exit(-1);
    }
    net.CreatePredictEnvironment(thrd_num);

    vector<InstancePtr> instances;
    LoadInstances(predict_in_file_name, net.GetFldNum(), instances);

    int_t cur_inst = 0;
    pthread_mutex_t cur_inst_mutex;
    pthread_mutex_init(&cur_inst_mutex, NULL);
    vector<vector<real_t> > feats(instances.size());

    vector<pthread_t> threads(thrd_num);
    vector<GetFeatThreadParam> thread_params(thrd_num);
    for (i = 0; i < thrd_num; ++i){
        thread_params[i].net = &net;
        thread_params[i].instances = &instances;
        thread_params[i].feats = &feats;
        thread_params[i].cur_inst = &cur_inst;
        thread_params[i].cur_inst_mutex = &cur_inst_mutex;
        thread_params[i].thrd_id = i;
        pthread_create(&threads[i], 0, &getFeatThread,
            reinterpret_cast<void *>(&thread_params[i]));
    }
    for (i = 0; i < thrd_num; ++i) {
        pthread_join(threads[i], 0);
    }

    ret = SaveFeatResult(feat_out_file_name, feats);
    if (!ret) {
        fprintf(stderr, "Save Feat result failed.\n");
        exit(-1);
    }
}

int main(int_t argc, char **argv) {
#if defined(FDCNN_PREDICT)
    Predict(argc, argv);
#elif defined(FDCNN_GETFEAT)
    GetFeat(argc, argv);
#elif defined(FDCNN_SWITCH_MODEL)
    SwitchModel(argc, argv);
#else
    Train(argc, argv);
#endif
}

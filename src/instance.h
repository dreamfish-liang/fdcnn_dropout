/*==============================================================================
 *   Copyright (C) 2016 All rights reserved.
 *
 *  File Name   : instance.h
 *  Author      : Zhongping Liang
 *  Date        : 2016-07-05
 *  Version     : 1.0
 *  Description : This file provides declarations for instance.
 *============================================================================*/

#ifndef DCNN_INSTANCE_H_
#define DCNN_INSTANCE_H_

#include <string>
#include <vector>
#include <tr1/memory>

#include "dcnn_utils.h"

namespace fdcnn
{

/*
 * struct Instance.
 *  The input instance of dcnn, saving features and label for a sentence.
 *  Each token in a sentence can be represent as nflds feature. The member
 *  variable features stores these features. The size of features is equal
 *  to nflds(number of fields), and the size of each element in features is
 *  euqal to the length of the sentence.
 */
struct Instance
{
    Instance(int_t nflds) : features(nflds), label() { }
    ~Instance() { }
    std::vector<std::vector<std::string> > features;//1 feat pos, 2 sen pos
    std::string label;
    /*
     *  @brief      This func gets the length of sentence.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @return     the length of sentence.
     */
    inline int_t GetSenLen() {
        return static_cast<int_t>(features.front().size());
    }

    /*
     *  @brief      This func gets the number of fields.
     *  @author     Zhongping Liang
     *  @date       2016-07-05
     *  @return     the number of fields.
     */
    inline int_t GetFldNum() {
        return static_cast<int_t>(features.size());
    }
};

typedef std::tr1::shared_ptr<Instance> InstancePtr;

} // namespace fdcnn

#endif // DCNN_INSTANCE_H_

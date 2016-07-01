#ifndef DATA_LOADER_H_
#define DATA_LOADER_H_

#include <string>
#include <vector>
#include <tr1/memory>

#include "dcnn_utils.h"

struct Instance
{
    Instance(int_t nflds) : features(nflds), label() { }
    ~Instance() { }
    std::vector<std::vector<std::string> > features;//1 feat pos, 2 sen pos
    std::string label;
    inline int_t GetSenLen()
    {
        return static_cast<int_t>(features.front().size());
    }
    inline int_t GetFldNum()
    {
        return static_cast<int_t>(features.size());
    }
};

typedef std::tr1::shared_ptr<Instance> InstancePtr;


#endif // DATA_LOADER_H_

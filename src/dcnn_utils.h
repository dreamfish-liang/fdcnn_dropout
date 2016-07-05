/*==============================================================================
 *   Copyright (C) 2016 All rights reserved.
 *
 *  File Name   : dcnn_util.h
 *  Author      : Zhongping Liang
 *  Date        : 2016-07-05
 *  Version     : 1.0
 *  Description : This file provides some utils for dcnn.
 *============================================================================*/

#ifndef DCNN_UTILS_H_
#define DCNN_UTILS_H_

#include <string>
#include <fstream>
#include <pthread.h>
#include <cstdio>
#include <iostream>
#include <sstream>

namespace fdcnn
{

typedef double real_t;
typedef int32_t int_t;
typedef uint32_t uint_t;
typedef uint64_t ulong_t;

const int_t MAX_SEN_LEN = 10000;
const int_t MAX_WORD_LEN = 100;
const int_t MAX_FIELD_NUM = 100;
const int_t LOG_PER_EXA = 1000;
const int_t BUFF_SIZE = 64;
//const std::string NIL = "NIL";
const std::string NIL = "__NIL__";

/*
 * struct ScopedLock.
 *  ScopedLock auto release lock.
 */
struct ScopedLock
{
public:
    ScopedLock(pthread_mutex_t &in) : mut(in){ pthread_mutex_lock(&mut); }
    ~ScopedLock() { pthread_mutex_unlock(&mut); }
private :
    pthread_mutex_t &mut;
};

/*
 * struct ScopedFstream.
 *   ScopedFstream auto close fstream.
 */
template <typename T>
struct ScopedFstream
{
public :
    ScopedFstream(T &in) : file(in) { }
    ~ScopedFstream() { if (file.is_open()) { file.close(); } }
private:
    T &file;
};

/*
 * struct ScopedFile.
 *  ScopedFile auto close file.
 */
struct ScopedFile
{
public :
    ScopedFile(FILE *in) : file(in){ }
    ~ScopedFile() { fclose(file); }
private :
    FILE * file;
};

/*
 * enum LogLevel.
 */
enum LogLevel {
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_FATAL
};

const LogLevel gLogLevel = LOG_LEVEL_INFO;
//const LogLevel gLogLevel = LOG_LEVEL_DEBUG;

/*
 *  @brief      This func gets level C-type string.
 *  @author     Zhongping Liang
 *  @date       2016-07-05
 *  @param      lvl: the log level.
 *  @return     the level C-type string.
 */
inline const char *GetLogLevelStr(LogLevel lvl)
{
    switch (lvl)
    {
    case LOG_LEVEL_DEBUG:
        return "[DEBUG]";
    case LOG_LEVEL_INFO:
        return "[INFO]";
    case LOG_LEVEL_WARN:
        return "[WARN]";
    case LOG_LEVEL_ERROR:
        return "[ERROR]";
    case LOG_LEVEL_FATAL:
        return "[FATAL]";
    default:
        break;
    }
    return "";
}

/*
 * struct OutBuffer.
 *  Collect the log output buffer.
 */
struct OutBuffer
{
    template <typename T>
    OutBuffer& operator()(T message) {
        oss << message;
        return *this;
    }
    std::ostringstream oss;
    inline std::string str() { return oss.str(); }
};

#define LOG(level, message) \
if (level >= gLogLevel) { \
    time_t timer; \
    time(&timer); \
    char stime[BUFF_SIZE]; \
    strftime(stime, BUFF_SIZE, "[%Y-%m-%d %X]", localtime(&timer)); \
    OutBuffer out_buff; \
    out_buff(stime)('\t')(GetLogLevelStr(level))('\t')message('\n'); \
    std::cout << out_buff.str(); \
}

#define LOG_DEBUG(message) LOG(LOG_LEVEL_DEBUG, message)
#define LOG_INFO(message)  LOG(LOG_LEVEL_INFO,  message)
#define LOG_WARN(message)  LOG(LOG_LEVEL_WARN,  message)
#define LOG_ERROR(message) LOG(LOG_LEVEL_ERROR, message)
#define LOG_FATAL(message) LOG(LOG_LEVEL_FATAL, message)

} // namespace fdcnn

#endif //DCNN_UTILS_H_

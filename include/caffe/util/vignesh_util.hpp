#ifndef CAFFE_UTIL_VIGNESH_UTIL_H_
#define CAFFE_UTIL_VIGNESH_UTIL_H_

#include <string>
#include <cstdarg>

using namespace std;

namespace caffe {

// stringprintf
inline std::string stringprintf(const char* fmt, ...){
    int size = 512;
    char* buffer = 0;
    buffer = new char[size];
    va_list vl;
    va_start(vl, fmt);
    int nsize = vsnprintf(buffer, size, fmt, vl);
    if(size<=nsize){ //fail delete buffer and try again
        delete[] buffer;
        buffer = 0;
        buffer = new char[nsize+1]; //+1 for /0
        nsize = vsnprintf(buffer, size, fmt, vl);
    }
    std::string ret(buffer);
    va_end(vl);
    delete[] buffer;
    return ret;
}

inline vector<string> strsplit(string str, string delim) { 
  int start = 0;
  int end; 
  vector<string> v; 
  while( (end = str.find(delim, start)) != string::npos )
  { 
        v.push_back(str.substr(start, end-start)); 
        start = end + delim.length(); 
  } 
  v.push_back(str.substr(start)); 
  return v; 
}

}

#endif

#include <vector>
#include <iostream>
#include <chrono>
#include <thread>

#if defined(_WIN32)

#else
#include <unistd.h>
#endif

#include <cstring>
#include <string>

#define getUintD(b) *(uint32_t*)b
#define getUint64D(b) *(uint64_t*)b
#define writeUintD(b,v) std::memcpy(b,&v,sizeof(uint32_t))
#define writeUint64D(b,v) std::memcpy(b,&v,sizeof(uint64_t))


extern bool silent;
extern bool presetArgs;

typedef uint64_t* uint256;

uint256 uint256_new();
void uint256_shift_left(uint256 value, uint8_t shift);
void uint256_set(uint256 out, uint64_t value);
void uint256_set_compact(uint256 out, uint32_t compact);
void uint256_set_bytes(uint256 out, uint8_t* bytes);
int8_t uint256_compare(uint8_t* left, uint8_t* right);

void waitMs(int ms);

std::chrono::time_point<std::chrono::high_resolution_clock> getTime();

int getMs(std::chrono::time_point<std::chrono::high_resolution_clock> t1, std::chrono::time_point<std::chrono::high_resolution_clock> t2);

bool setThreadAffinity(uint64_t cpu_id);

unsigned int getUint(unsigned char* b);

unsigned long long getUint64(unsigned char* b);

void writeUint(unsigned char* b, unsigned int v);

void writeUint64(unsigned char* b, uint64_t v);

void exitMsg(std::string msg);

int tryParseInt(std::string s);
double tryParseFloat(std::string s);

void miner_log(std::string tag, std::string message, std::string color = "def");

std::vector<std::string> split(std::string s, std::string delim);

struct Arg {
    std::string name;
    std::string val;
};

class CmdArgs {

public:
    CmdArgs(char* argv[], int argc, bool presetArgs, bool exitError);
    CmdArgs();

    std::string get(std::string arg_name);

    void add(std::string name, std::string val);

    std::string get(int index);

    Arg getFull(int index);

    int len();
    bool error;
    std::vector<Arg> m_args;
private:
};

struct CpuInfo {
    int cores;
    std::vector<int> affinity;
    std::string cpuName;
};

CpuInfo getCpuInfo();

void toHex(const unsigned char* input, char* output, size_t inputLength);
std::string toHex32(const uint32_t& input);
std::string toHex(const uint8_t* input, size_t inputLength);
std::string toHex(const std::vector<uint8_t>& input);
std::string toHex(const uint64_t& input);
void fromHex(unsigned char* input, int l, unsigned char* output);
std::string reverseBlob(std::string s);
std::vector<uint8_t> fromHex(const std::string& input);

uint64_t getBigNonceMask(int nonceSize, int i);
std::string getBigNum(const unsigned char* v, int l);

//void memcpy_avx2(char* dst, const char* src, size_t size);


//using nvidia_ctx = struct nvidia_ctx;


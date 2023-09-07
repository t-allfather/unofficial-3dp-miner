#include "utils.h"
#include "main.h"
#include <x86intrin.h>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include <Windows.h>
#include <intrin.h>
#endif

#include <bitset>

bool silent = false;
void waitMs(int ms) {
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

std::chrono::time_point<std::chrono::high_resolution_clock> getTime() {
	auto t = std::chrono::high_resolution_clock::now();
	return t;
}

unsigned int getUint(unsigned char* b) {
	return *reinterpret_cast<uint32_t*>(b);
}

unsigned long long getUint64(unsigned char* b) {
	return *reinterpret_cast<uint64_t*>(b);
}

void writeUint(unsigned char* b, uint32_t v) {
	std::memcpy(b, &v, sizeof(uint32_t));
}

void writeUint64(unsigned char* b, uint64_t v) {
	std::memcpy(b, &v, sizeof(uint64_t));
}

int getMs(std::chrono::time_point<std::chrono::high_resolution_clock> t1, std::chrono::time_point<std::chrono::high_resolution_clock> t2) {
	auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
	return ms_int.count();
}

void miner_log(std::string tag, std::string message, std::string color) {
	if (!silent) {
		miner_log_main_thread(tag, message, color);
	}
}

bool setThreadAffinity(uint64_t cpu_id)
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	if (cpu_id == -1) {
		return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(0xffffffffffffffff));
	}
	else if (cpu_id == -2) {
		return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(0xffffffff00000000));
	}
	else if (cpu_id == -3) {
		return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(0x00000000ffffffff));
	}
	bool result = 0;
	result = SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR((unsigned long long)(1) << cpu_id));
#else
	cpu_set_t mn;
	CPU_ZERO(&mn);

	if (cpu_id == -1) {
		for (int i = 0; i < 64; i++) {
			CPU_SET(i, &mn);
		}
	}
	else {
		CPU_SET(cpu_id, &mn);
	}

#   ifndef __ANDROID__
	const bool result = (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mn) == 0);
#   else
	const bool result = (sched_setaffinity(gettid(), sizeof(cpu_set_t), &mn) == 0);
#   endif
#endif`
	std::this_thread::sleep_for(std::chrono::milliseconds(1));
	return result;
}

void exitMsg(std::string msg) {
	if (!silent)
		std::cout << msg << std::endl;
	exit(0);
}

int tryParseInt(std::string s) {
	int v = atoi(s.c_str());
	return v;
}

double tryParseFloat(std::string s) {
	double v = atof(s.c_str());
	return v;
}

std::vector<std::string> split(std::string s, std::string delim) {
	std::vector<std::string> o;
	auto start = 0U;
	auto end = s.find(delim);
	while (end != std::string::npos)
	{
		o.push_back(s.substr(start, end - start));
		start = end + delim.length();
		end = s.find(delim, start);
	}

	o.push_back(s.substr(start, end));
	return o;
}

CmdArgs::CmdArgs(){

}

CmdArgs::CmdArgs(char* argv[], int argc, bool presetArgs, bool exitError) {
	error = false;
	int i = 1;
	if (presetArgs) {

	}
	else {
		while (i < argc) {
			std::string s(argv[i]);
			std::string arg_name = "";
			if (s.rfind("--", 0) == 0) {
				for (int j = 2; j < s.size(); j++) {
					arg_name += s[j];
				}
			}
			else {
				if (exitError)
					exitMsg("Error parsing arguments: argument name required");
				else
					error = true;
			}
			if (!(i < argc - 1)) {
				std::string arg_val = "";
				m_args.push_back({ arg_name,arg_val });
				i++;
				continue;
			}
			std::string s2(argv[i + 1]);
			if (s2.rfind("--", 0) == 0) {
				std::string arg_val = "";
				m_args.push_back({ arg_name,arg_val });
				i++;
				continue;
			}
			i++;
			std::string arg_val = argv[i];
			m_args.push_back({ arg_name,arg_val });
			i++;
		}
	}
}

std::string CmdArgs::get(std::string arg_name) {
	for (int i = 0; i < m_args.size(); i++) {
		if (m_args[i].name == arg_name) {
			return m_args[i].val;
		}
	}
	return "NULL_ARG";
}

std::string CmdArgs::get(int index) {
	return m_args[index].val;
}

Arg CmdArgs::getFull(int index) {
	return m_args[index];
}

void CmdArgs::add(std::string name, std::string val) {
	m_args.push_back({ name,val });
}

int CmdArgs::len() {
	return m_args.size();
}

void cpuid(int* v, int f) {
#ifdef _WIN32
	__cpuid(v, (int)f);
#else
	asm volatile
		("cpuid" : "=a" (v[0]), "=b" (v[1]), "=c" (v[2]), "=d" (v[3])
			: "a" (f), "c" (0));
#endif
}

CpuInfo getCpuInfo() {
	CpuInfo info;
	int logical_cores = std::thread::hardware_concurrency();
	info.cores = logical_cores;
	for (int i = 0; i < logical_cores; i++) {
		info.affinity.push_back(i);
	}
	int CPUInfo[4] = { -1 };
	char* CPUBrandString = new char[0x40];
	cpuid(CPUInfo, 0x80000000);
	unsigned int nExIds = CPUInfo[0];
	memset(CPUBrandString, 0, sizeof(CPUBrandString));
	for (int i = 0x80000000; i <= nExIds; ++i)
	{
		cpuid(CPUInfo, i);
		if (i == 0x80000002)
			memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000003)
			memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
		else if (i == 0x80000004)
			memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
	}
	std::string cpuName = "";
	int x = strlen(CPUBrandString);
	while (x > 0 && (CPUBrandString[x] == ' ' || CPUBrandString[x] == 0)) {
		x--;
	}
	for (int i = 0; i <= x; i++) {
		cpuName += CPUBrandString[i];
	}
	info.cpuName = cpuName;
	delete CPUBrandString;
	return info;
}

void toHex(const unsigned char* input, char* output, size_t inputLength)
{
	char hexval[16] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

	for (int i = 0; i < inputLength; i++)
	{
		output[i * 2] = hexval[((input[i] >> 4) & 0xF)];
		output[(i * 2) + 1] = hexval[(input[i]) & 0x0F];
	}
}

void fromHex(unsigned char* input, int l, unsigned char* output) {
	for (int i = 0; i < l / 2; i++) {
		output[i] = 16 * (input[i * 2] - '0') + input[i * 2 + 1] - '0';
	}
}

std::string toHex32(const uint32_t& input)
{
	std::string result;
	result.resize(sizeof(input) * 2);

	toHex(reinterpret_cast<const unsigned char*>(&input), &result[0], sizeof(input));

	return result;
}

std::string toHex(const uint64_t& input)
{
	std::string result;
	result.resize(sizeof(input) * 2);

	toHex(reinterpret_cast<const unsigned char*>(&input), &result[0], sizeof(input));

	return result;
}

std::string toHex(const uint8_t* input, size_t inputLength)
{
	std::string result;
	result.resize(inputLength * 2);

	toHex(input, &result[0], inputLength);

	return result;
}

std::string toHex(const std::vector<uint8_t>& input)
{
	std::string result;
	result.resize(input.size() * 2);

	toHex(input.data(), &result[0], input.size());

	return result;
}

std::string reverseBlob(std::string s) {
	std::string o = "";
	for (int i = s.size() - 1; i >= 0; i -= 2) {
		o += s[i - 1];
		o += s[i];
	}
	return o;
}

int char2int(char input)
{
	if (input >= '0' && input <= '9')
	{
		return input - '0';
	}

	if (input >= 'A' && input <= 'F')
	{
		return input - 'A' + 10;
	}

	if (input >= 'a' && input <= 'f')
	{
		return input - 'a' + 10;
	}

	return -1;
}

std::vector<uint8_t> fromHex(const std::string& input)
{
	const size_t outputLength = input.size() / 2;

	std::vector<uint8_t> output(outputLength);

	for (int i = 0; i < outputLength; i++)
	{
		output[i] = char2int(input[i * 2]) * 16
			+ char2int(input[(i * 2) + 1]);
	}

	return output;
}

uint256 uint256_new() {
	uint256 out = (uint256)malloc(32);
	memset(out, 0, 32);
	return out;
}

void uint256_shift_left(uint256 value, uint8_t shift) {
	while (shift > 64) {
		for (int i = 0; i < 3; ++i) value[i] = value[i + 1];
		value[3] = 0;
		shift -= 64;
	}
	if (shift == 0) return;
	for (int i = 0; i < 3; ++i) {
		value[i] <<= shift;
		value[i] |= value[i + 1] >> (64 - shift);
	}
	value[3] <<= shift;
	return;
}

void uint256_set(uint256 out, uint64_t value) {
	out[3] = value;
}

void uint256_set_compact(uint256 out, uint32_t compact) {
	uint256_set(out, compact & 0xffffff);
	uint256_shift_left(out, (8 * ((compact >> 24) - 3)));
}

void uint256_set_bytes(uint256 out, uint8_t* bytes) {
	uint8_t* out8 = (uint8_t*)out;
	for (int i = 0; i < 32; i++) {
		out8[i] = out[i];
	}
}

int8_t uint256_compare(uint8_t* left, uint8_t* right) {
	for (int i = 0; i < 32; ++i) {
		if (left[i] < right[i]) return -1;
		if (left[i] > right[i]) return 1;
	}
	return 0;
}

inline uint16_t bswap_16(uint16_t x) {
	return (x >> 8) | (x << 8);
}

inline uint32_t bswap_32(uint32_t x) {
	return (bswap_16(x & 0xffff) << 16) | (bswap_16(x >> 16));
}

inline uint64_t bswap_64(uint64_t x) {
	return (((uint64_t)bswap_32(x & 0xffffffffull)) << 32) | (bswap_32(x >> 32));
}

uint64_t getBigNonceMask(int nonceSize, int i) {
	if (i == 0) {
		if (nonceSize == 1) {
			return 0x00FFFFFFFFFFFFFFULL;
		}
		else if (nonceSize == 2) {
			return 0x0000FFFFFFFFFFFFULL;
		}
		else if (nonceSize == 3) {
			return 0x000000FFFFFFFFFFULL;
		}
		else if (nonceSize == 4) {
			return 0x000000FFFFFFFFFFULL;
		}
		return 0;
	}
	else {
		if (nonceSize == 1) {
			return 0xFF00000000000000ULL;
		}
		else if (nonceSize == 2) {
			return 0xFFFF000000000000ULL;
		}
		else if (nonceSize == 3) {
			return 0xFFFFFF0000000000ULL;
		}
		else if (nonceSize == 4) {
			return 0xFFFFFFFF00000000ULL;
		}
		return 0;
	}
}

/*
void memcpy_avx2(char* dst, const char* src, size_t size)
{
	while(size) {
		_mm256_store_si256 ((__m256i*)dst, _mm256_load_si256((__m256i const*)src));
		src += 32;
		dst += 32;
		size -= 32;
	}
}
*/
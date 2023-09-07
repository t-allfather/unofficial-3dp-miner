#include <string>

#define PROTO_SOLO 0
#define PROTO_DEV 1
#define PROTO_POOL 2

struct miner_log_item {
	std::string tag;
	std::string text;
	std::string color = "def";
};

void miner_log_main_thread(std::string tag, std::string text, std::string color);
void mainAddThread(void(*f)());

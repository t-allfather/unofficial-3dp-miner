#include <vector>
#include <string>
#include <iostream>
#include <map>


struct JsonValue {
	int type = 0;
	std::vector<std::string> vals;
};

class Json {
public:
	Json(std::string data);
	std::string get(std::string name);
	std::vector<std::string> getArr(std::string name);
private:
	std::map<std::string, JsonValue> m_fields;
};
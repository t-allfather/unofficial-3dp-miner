#ifndef HTTP_HEDER
#define HTTP_HEADER

#include <vector>
#include <string>


class HttpHeader {
public:
    HttpHeader(std::string r);
    HttpHeader(std::vector<unsigned char> r);
    int resp_code();
    std::string data();
    std::vector<unsigned char> rawData();
    bool ok();
private:
    std::vector<unsigned char> raw;
    std::vector<std::string> lines;
};

HttpHeader HttpRequest(std::string host, std::string method, std::string path, std::string data, std::string cookie, std::string header,std::string auth);
std::string EncodeBase64(const std::string& str);
void initTcp();
#endif // !HTTP_HEDER
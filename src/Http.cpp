#include "Http.h"
#include <ostream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include "utils.h"
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#ifndef WS_LIB
#define WS_LIB
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment (lib, "Ws2_32.lib")
#endif
#else

#include <strings.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>

#endif

std::string ctostr(char x)
{
	std::string c(1, x);
	if (x == ' ') c = " ";
	else if (x == '\n') c = "\n";
	else if (x == '\r') c = "\r";
	return c;
}

std::vector<std::string> split(std::string s, char c) {
	std::vector<std::string> p;
	std::string w = "";
	for (int i = 0; i < (int)s.size(); i++) {
		if (s[i] != c) {
			w = w + ctostr(s[i]);
		}
		else {
			p.push_back(w);
			w = "";
		}
	}
	if (w != "") p.push_back(w);
	return p;
}

HttpHeader::HttpHeader(std::string r) {
	std::vector<unsigned char> d;
	for (int i = 0; i < r.size(); i++) {
		d.push_back(r[i]);
	}
	raw = d;
	lines = split(r, '\n');
}

HttpHeader::HttpHeader(std::vector<unsigned char> r) {
	raw = r;
	std::string d = "";
	for (int i = 0; i < r.size(); i++) {
		d += r[i];
	}
	lines = split(d, '\n');
	/*
	for (int i = 0; i < raw.size(); i++) {
		std::cout << raw[i];
	}
	std::cout << std::endl << std::endl;
	std::cout << "end\n";
	*/
}

int HttpHeader::resp_code() {
	std::vector<std::string> s = split(lines[0], ' ');
	return strtol(s[1].c_str(), 0, 10);
}

bool HttpHeader::ok() {
	if (raw.size() < 20) {
		std::string s = "";
		for (int i = 0; i < raw.size(); i++) {
			s += raw[i];
		}
		if (s == "CONNECTION_ERROR") {
			return false;
		}
		return true;
	}
	return true;
}

std::vector<unsigned char> HttpHeader::rawData() {
	std::vector<unsigned char> d;
	int clinelen = 0;
	int s = 0;
	for (int i = 0; i < raw.size(); i++) {
		if (raw[i] == '\n') {
			if (clinelen == 1) {
				if (raw[i - 1] == '\r') {
					s = i + 1;
					break;
				}
			}
			clinelen = 0;
		}
		else {
			clinelen++;
		}
	}
	for (int i = s; i < raw.size(); i++) {
		d.push_back(raw[i]);
	}
	return d;
}

std::string HttpHeader::data() {
	std::string d = "";
	int clinelen = 0;
	int s = 0;
	for (int i = 0; i < raw.size(); i++) {
		if (raw[i] == '\n') {
			if (clinelen == 1) {
				if (raw[i - 1] == '\r') {
					s = i + 1;
					break;
				}
			}
			clinelen = 0;
		}
		else {
			clinelen++;
		}
	}
	for (int i = s; i < raw.size(); i++) {
		d += raw[i];
	}
	return d;
}

std::string url_encode(std::string& value) {
	std::ostringstream escaped;
	escaped.fill('0');
	escaped << std::hex;
	for (std::string::const_iterator i = value.begin(), n = value.end(); i != n; ++i) {
		std::string::value_type c = (*i);
		if (isalnum(c) || c == '-' || c == '?' || c == '_' || c == '.' || c == '~' || c == '/' || c == '&' || c == '=') {
			escaped << c;
			continue;
		}
		escaped << std::uppercase;
		escaped << '%' << std::setw(2) << int((unsigned char)c);
		escaped << std::nouppercase;
	}

	return escaped.str();
}


int initialized = 0;
void initTcp() {

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	WSADATA wsaData;
	int iResult;
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0) {
		printf("WSAStartup failed: %d\n", iResult);
		return;
	}
#endif
	initialized = 1;
}

std::string EncodeBase64(const unsigned char* pch, size_t len)
{
	static const char* pbase64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	std::string strRet = "";
	strRet.reserve((len + 2) / 3 * 4);

	int mode = 0, left = 0;
	const unsigned char* pchEnd = pch + len;

	while (pch < pchEnd)
	{
		int enc = *(pch++);
		switch (mode)
		{
		case 0: // we have no bits
			strRet += pbase64[enc >> 2];
			left = (enc & 3) << 4;
			mode = 1;
			break;

		case 1: // we have two bits
			strRet += pbase64[left | (enc >> 4)];
			left = (enc & 15) << 2;
			mode = 2;
			break;

		case 2: // we have four bits
			strRet += pbase64[left | (enc >> 6)];
			strRet += pbase64[enc & 63];
			mode = 0;
			break;
		}
	}

	if (mode)
	{
		strRet += pbase64[left];
		strRet += '=';
		if (mode == 1)
			strRet += '=';
	}

	return strRet;
}

std::string EncodeBase64(const std::string& str) { return EncodeBase64((const unsigned char*)str.c_str(), str.size()); }

HttpHeader HttpRequest(std::string host, std::string method, std::string path, std::string data, std::string cookie, std::string header,std::string auth="") {
	if (initialized == 0)
		initTcp();
	path = url_encode(path);
	if (header == "") {
		header += method + " " + path + " HTTP/1.1\r\n";
		header += "Host: " + host + "\r\n";
		header += "User-Agent: custom-http-agent\r\n";
		header += "Accept: */*\r\n";
        header += "Content-Type: application/json\r\n";
		header += "Content-Length: " + std::to_string(data.size()) + "\r\n";
		header += "Connection: close\r\n";
		if (cookie != "") {
			header += "Cookie: " + cookie + "\r\n";
		}
		if (auth != "") {
			header += "Authorization: Basic "+ EncodeBase64(auth) +"\r\n";
		}
		header += "\r\n";
		header += data;
	}
	std::string _host;
	int _port;
	std::vector<std::string> _split = split(host, ':');
	_host = _split[0];
	_port = tryParseInt(_split[1]);
	struct sockaddr_in serv_addr;
#ifdef _WIN32
	SOCKET _sock = 0;
#else
	int _sock = 0;
#endif
	if ((_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
#ifdef WIN32
		closesocket(_sock);
#else
		shutdown(_sock, SHUT_RDWR);
		close(_sock);
#endif
		printf("[Http request] Socket creation error \n");
		return HttpHeader("CONNECTION_ERROR");
	}

	struct timeval tv;
    int timeout = 5;
    tv.tv_sec = timeout;
    tv.tv_usec = 0;
    setsockopt(_sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
	setsockopt(_sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(_port);

	int x = 0;
	for (int i = 0; i < _host.size(); i++) {
		if (_host[i] == '.') {
			x++;
		}
	}

	if (x <= 2) {
		struct hostent* ghbn = gethostbyname(_host.c_str());
		const char* s = inet_ntoa(*(struct in_addr*)ghbn->h_addr);
		if (inet_pton(AF_INET, s, &serv_addr.sin_addr) <= 0) {
			printf("[Http request] Invalid address/ Address not supported \n");
#ifdef WIN32
			closesocket(_sock);
#else
			shutdown(_sock, SHUT_RDWR);
			close(_sock);
#endif
			return HttpHeader("CONNECTION_ERROR");
		}
	}
	else {
		if (inet_pton(AF_INET, _host.c_str(), &serv_addr.sin_addr) <= 0) {
			printf("[Http request] Invalid address/ Address not supported \n");
#ifdef WIN32
			closesocket(_sock);
#else
			shutdown(_sock, SHUT_RDWR);
			close(_sock);
#endif
			return HttpHeader("CONNECTION_ERROR");
		}
	}

	int status = connect(_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
	if (status < 0) {
#ifdef WIN32
		closesocket(_sock);
#else
		shutdown(_sock, SHUT_RDWR);
		close(_sock);
#endif
		return HttpHeader("CONNECTION_ERROR");
	}
	else {
		const char* dt = (header.c_str());
		int snt = 0;
		int p =0;
		while (true)
		{
			int min_ = 1024*8;
			if(min_ > header.size() -p){
				min_ = header.size() -p;
			}
			int _s = send(_sock, dt+p, min_, 0);
			if(_s<1){
				snt = -1;
				break;
			}
			p+=_s;
			snt += _s;
			if(p >= header.size()){
				break;
			}
		}
		
		if (snt < 0)
		{
#ifdef WIN32
			closesocket(_sock);
#else
			shutdown(_sock, SHUT_RDWR);
			close(_sock);
#endif
			return HttpHeader("CONNECTION_ERROR");
		}
		else {
			unsigned char* buffer = new unsigned char[1024 * 16];
			std::vector<unsigned char> r;
			int _r = 1;
			while (_r) {
#ifdef WIN32
				_r = recv(_sock, (char*)buffer, 1024 * 16, 0);
#else
				_r = read(_sock, buffer, 1024 * 16);
#endif
				if (_r < 1) break;
				for (int j = 0; j < _r; j++) {
					r.push_back(buffer[j]);
				}
			}
			delete[]buffer;
#ifdef WIN32
			closesocket(_sock);
#else
			shutdown(_sock, SHUT_RDWR);
			close(_sock);
#endif
			return HttpHeader(r);
		}
	}
}

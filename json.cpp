#include "json.h"

Json::Json(std::string data) {
	int depth = 0;
	for (int i = 0; i < data.size(); i++) {
		if (data[i] == '"') {
			i++;
			std::string s = "";
			std::string s2 = "";
			while (i < data.size() && data[i] != '"') {
				s += data[i];
				i++;
			}
			while (i < data.size() && data[i] != ':')
				i++;
			i++;
			if (i >= data.size()) {
				break;
			}
			char cc = data[i];
			if (data[i] == '"') {
				i++;
				while (i < data.size() && data[i] != '"') {
					s2 += data[i];
					i++;
				}
			}
			else if (data[i] == '{') {
				i++;
				while (i < data.size() && data[i] != '}') {
					s2 += data[i];
					i++;
				}
			}
			else if (data[i] == '[') {
				while (i < data.size() && data[i] != ']') {
					s2 += data[i];
					i++;
				}
				if (i < data.size()) {
					s2 += data[i];
				}
			}
			else {
				while (i < data.size() && data[i] != ' ' && data[i] != ',' && data[i] != '}') {
					s2 += data[i];
					i++;
				}
			}
			if (s2.size() > 1 && s2[0] == '[' && s2[s2.size() - 1] == ']') {
				std::vector<std::string> s3;
				std::string w = "";
				for (int j = 1; j < s2.size() - 1; j++) {
					if (s2[j] == ',') {
						if (w.size() == 0)
							break;
						if (w[0] == '"' && w[w.size() - 1] == '"') {
							std::string ww = "";
							for (int q = 1; q < w.size() - 1; q++) {
								ww += w[q];
							}
							w = ww;
						}
						s3.push_back(w);
						w = "";
					}
					else {
						w += s2[j];
					}
				}
				if (w != "") {
					if (w[0] == '"' && w[w.size() - 1] == '"') {
						std::string ww = "";
						for (int q = 1; q < w.size() - 1; q++) {
							ww += w[q];
						}
						w = ww;
					}
					s3.push_back(w);
				}
				JsonValue t;
				t.type = 1;
				t.vals = s3;
				m_fields.insert({ s,t });
			}
			else {
				std::vector<std::string> v;
				v.push_back(s2);
				JsonValue t;
				t.type = 1;
				t.vals = v;
				m_fields.insert({ s,t });
			}
			//std::cout << s << "@" << s2 << "\n";
		}
	}
}

std::string Json::get(std::string name) {
	if (m_fields.find(name) == m_fields.end()) {
		return "NULL";
	}
	else {
		return m_fields[name].vals[0];
	}
}

std::vector<std::string> Json::getArr(std::string name) {
	if (m_fields.find(name) == m_fields.end()) {
		std::vector<std::string> v;
		return v;
	}
	else {
		return m_fields[name].vals;
	}
}

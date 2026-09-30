// Minimal JSON object builder, enough for meta.json and summaries. No external dependencies.
// Usage: sb::Json j; j.add("gpu", name).add("sm", 28).add_raw("fit", other.str()); fputs(j.str().c_str(), f);
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace sb {

inline std::string json_escape(const std::string &s) {
    std::string o;
    o.reserve(s.size() + 2);
    o += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof b, "\\u%04x", c);
                    o += b;
                } else {
                    o += (char)c;
                }
        }
    }
    o += '"';
    return o;
}

inline std::string json_number(double v) {
    if (!std::isfinite(v)) return "null";
    char b[40];
    snprintf(b, sizeof b, "%.17g", v);
    return b;
}

class Json {
public:
    Json &add(const std::string &k, const std::string &v) { return raw(k, json_escape(v)); }
    Json &add(const std::string &k, const char *v) { return raw(k, v ? json_escape(v) : "null"); }
    Json &add(const std::string &k, bool v) { return raw(k, v ? "true" : "false"); }
    Json &add(const std::string &k, int v) { return raw(k, std::to_string(v)); }
    Json &add(const std::string &k, unsigned v) { return raw(k, std::to_string(v)); }
    Json &add(const std::string &k, long v) { return raw(k, std::to_string(v)); }
    Json &add(const std::string &k, unsigned long v) { return raw(k, std::to_string(v)); }
    Json &add(const std::string &k, long long v) { return raw(k, std::to_string(v)); }
    Json &add(const std::string &k, unsigned long long v) { return raw(k, std::to_string(v)); }
    Json &add(const std::string &k, double v) { return raw(k, json_number(v)); }
    Json &add(const std::string &k, const std::vector<double> &v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + json_number(v[i]);
        return raw(k, s + "]");
    }
    Json &add(const std::string &k, const std::vector<std::string> &v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + json_escape(v[i]);
        return raw(k, s + "]");
    }
    Json &add_null(const std::string &k) { return raw(k, "null"); }
    // v must already be valid JSON (object, array, number...).
    Json &add_raw(const std::string &k, const std::string &v) { return raw(k, v); }
    Json &add(const std::string &k, const Json &v) { return raw(k, v.str()); }

    std::string str() const { return "{" + body_ + "}"; }

private:
    Json &raw(const std::string &k, const std::string &v) {
        if (!body_.empty()) body_ += ",";
        body_ += json_escape(k) + ":" + v;
        return *this;
    }
    std::string body_;
};

}  // namespace sb

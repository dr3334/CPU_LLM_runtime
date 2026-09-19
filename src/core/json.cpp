#include "llmrt/json.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "llmrt/common.h"

namespace llmrt {
namespace json {

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------
namespace {

class Parser {
 public:
  explicit Parser(std::string_view text) : s_(text) {}

  Value parse_document() {
    skip_ws();
    Value v = parse_value();
    skip_ws();
    if (!eof()) fail("trailing characters after top-level value");
    return v;
  }

 private:
  std::string_view s_;
  size_t pos_ = 0;
  int line_ = 1;
  int col_ = 1;

  [[noreturn]] void fail(const std::string& msg) const {
    throw Error("json: " + msg + " at line " + std::to_string(line_) +
                ", column " + std::to_string(col_));
  }

  bool eof() const { return pos_ >= s_.size(); }
  char peek() const { return eof() ? '\0' : s_[pos_]; }

  char get() {
    const char c = s_[pos_++];
    if (c == '\n') {
      ++line_;
      col_ = 1;
    } else {
      ++col_;
    }
    return c;
  }

  void skip_ws() {
    while (!eof()) {
      const char c = peek();
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        get();
      } else {
        break;
      }
    }
  }

  void expect(char c) {
    if (eof() || peek() != c) {
      fail(std::string("expected '") + c + "'");
    }
    get();
  }

  bool consume(char c) {
    if (!eof() && peek() == c) {
      get();
      return true;
    }
    return false;
  }

  void expect_literal(const char* lit) {
    for (const char* p = lit; *p; ++p) {
      if (eof() || get() != *p) fail(std::string("invalid literal, expected \"") + lit + "\"");
    }
  }

  Value parse_value() {
    if (eof()) fail("unexpected end of input");
    switch (peek()) {
      case '{': return parse_object();
      case '[': return parse_array();
      case '"': return Value::make_string(parse_string());
      case 't': expect_literal("true"); return Value::make_bool(true);
      case 'f': expect_literal("false"); return Value::make_bool(false);
      case 'n': expect_literal("null"); return Value::make_null();
      default: return parse_number();
    }
  }

  Value parse_object() {
    expect('{');
    Value::Object obj;
    skip_ws();
    if (consume('}')) return Value::make_object(std::move(obj));
    while (true) {
      skip_ws();
      if (peek() != '"') fail("expected object key string");
      std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      obj.emplace_back(std::move(key), parse_value());
      skip_ws();
      if (consume(',')) continue;
      if (consume('}')) break;
      fail("expected ',' or '}' in object");
    }
    return Value::make_object(std::move(obj));
  }

  Value parse_array() {
    expect('[');
    Value::Array arr;
    skip_ws();
    if (consume(']')) return Value::make_array(std::move(arr));
    while (true) {
      skip_ws();
      arr.push_back(parse_value());
      skip_ws();
      if (consume(',')) continue;
      if (consume(']')) break;
      fail("expected ',' or ']' in array");
    }
    return Value::make_array(std::move(arr));
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (eof()) fail("unterminated string");
      const char c = get();
      if (c == '"') break;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (eof()) fail("unterminated escape sequence");
      switch (get()) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': parse_hex_surrogate(out); break;
        default: fail("invalid escape sequence");
      }
    }
    return out;
  }

  uint32_t parse_hex4() {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      if (eof()) fail("truncated \\u escape");
      const char c = get();
      v <<= 4;
      if (c >= '0' && c <= '9') {
        v |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        v |= static_cast<uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        v |= static_cast<uint32_t>(c - 'A' + 10);
      } else {
        fail("invalid hex digit in \\u escape");
      }
    }
    return v;
  }

  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  // Decodes a \uXXXX escape, pairing a UTF-16 high surrogate with the low
  // surrogate that must follow it, and appends the UTF-8 encoding to `out`.
  void parse_hex_surrogate(std::string& out) {
    const uint32_t hi = parse_hex4();
    if (hi < 0xD800 || hi > 0xDBFF) {
      if (hi >= 0xDC00 && hi <= 0xDFFF) fail("unexpected low surrogate");
      append_utf8(out, hi);
      return;
    }
    if (eof() || get() != '\\') fail("expected low surrogate");
    if (eof() || get() != 'u') fail("expected \\u for low surrogate");
    const uint32_t lo = parse_hex4();
    if (lo < 0xDC00 || lo > 0xDFFF) fail("invalid low surrogate");
    append_utf8(out, 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00));
  }

  Value parse_number() {
    const size_t start = pos_;
    bool is_int = true;

    consume('-');

    if (eof()) fail("unexpected end of number");
    if (peek() == '0') {
      get();
    } else if (std::isdigit(static_cast<unsigned char>(peek()))) {
      while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) get();
    } else {
      fail("invalid number");
    }

    if (consume('.')) {
      is_int = false;
      if (eof() || !std::isdigit(static_cast<unsigned char>(peek()))) {
        fail("expected digit after decimal point");
      }
      while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) get();
    }

    if (!eof() && (peek() == 'e' || peek() == 'E')) {
      is_int = false;
      get();
      if (!eof() && (peek() == '+' || peek() == '-')) get();
      if (eof() || !std::isdigit(static_cast<unsigned char>(peek()))) {
        fail("expected digits in exponent");
      }
      while (!eof() && std::isdigit(static_cast<unsigned char>(peek()))) get();
    }

    const std::string tok(s_.substr(start, pos_ - start));

    if (is_int) {
      errno = 0;
      char* end = nullptr;
      const long long v = std::strtoll(tok.c_str(), &end, 10);
      if (errno == 0 && end == tok.c_str() + tok.size()) {
        return Value::make_int(static_cast<int64_t>(v));
      }
      // Overflowed int64: fall back to double.
    }
    return Value::make_double(std::strtod(tok.c_str(), nullptr));
  }
};

void dump_impl(const Value& v, std::string& out, int indent, int depth);

void dump_indent(std::string& out, int indent, int depth) {
  if (indent > 0) out.append(static_cast<size_t>(indent) * depth, ' ');
}

void dump_string(const std::string& s, std::string& out) {
  out.push_back('"');
  for (const char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

void dump_impl(const Value& v, std::string& out, int indent, int depth) {
  switch (v.type()) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += v.as_bool() ? "true" : "false"; break;
    case Type::Number:
      if (v.is_integer()) {
        out += std::to_string(v.as_int());
      } else {
        std::ostringstream os;
        os << v.as_double();
        out += os.str();
      }
      break;
    case Type::String: dump_string(v.as_string(), out); break;
    case Type::Array: {
      const auto& a = v.as_array();
      if (a.empty()) { out += "[]"; break; }
      out += '[';
      for (size_t i = 0; i < a.size(); ++i) {
        if (i) out += ',';
        if (indent > 0) { out += '\n'; dump_indent(out, indent, depth + 1); }
        dump_impl(a[i], out, indent, depth + 1);
      }
      if (indent > 0) { out += '\n'; dump_indent(out, indent, depth); }
      out += ']';
      break;
    }
    case Type::Object: {
      const auto& o = v.as_object();
      if (o.empty()) { out += "{}"; break; }
      out += '{';
      for (size_t i = 0; i < o.size(); ++i) {
        if (i) out += ',';
        if (indent > 0) { out += '\n'; dump_indent(out, indent, depth + 1); }
        dump_string(o[i].first, out);
        out += indent > 0 ? ": " : ":";
        dump_impl(o[i].second, out, indent, depth + 1);
      }
      if (indent > 0) { out += '\n'; dump_indent(out, indent, depth); }
      out += '}';
      break;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------

Value Value::make_null() { return Value(); }

Value Value::make_bool(bool b) {
  Value v;
  v.type_ = Type::Bool;
  v.bool_ = b;
  return v;
}

Value Value::make_int(int64_t i) {
  Value v;
  v.type_ = Type::Number;
  v.int_ = i;
  v.double_ = static_cast<double>(i);
  v.is_int_ = true;
  return v;
}

Value Value::make_double(double d) {
  Value v;
  v.type_ = Type::Number;
  v.double_ = d;
  v.int_ = static_cast<int64_t>(d);
  v.is_int_ = false;
  return v;
}

Value Value::make_string(std::string s) {
  Value v;
  v.type_ = Type::String;
  v.string_ = std::move(s);
  return v;
}

Value Value::make_array(Array a) {
  Value v;
  v.type_ = Type::Array;
  v.array_ = std::move(a);
  return v;
}

Value Value::make_object(Object o) {
  Value v;
  v.type_ = Type::Object;
  v.object_ = std::move(o);
  return v;
}

bool Value::as_bool() const {
  LLMRT_CHECK(type_ == Type::Bool, std::string("json: expected bool, got ") + type_name(type_));
  return bool_;
}

int64_t Value::as_int() const {
  LLMRT_CHECK(type_ == Type::Number, std::string("json: expected number, got ") + type_name(type_));
  LLMRT_CHECK(is_int_, "json: number is not an integer");
  return int_;
}

double Value::as_double() const {
  LLMRT_CHECK(type_ == Type::Number, std::string("json: expected number, got ") + type_name(type_));
  return double_;
}

double Value::as_number() const {
  LLMRT_CHECK(type_ == Type::Number, std::string("json: expected number, got ") + type_name(type_));
  return double_;
}

const std::string& Value::as_string() const {
  LLMRT_CHECK(type_ == Type::String, std::string("json: expected string, got ") + type_name(type_));
  return string_;
}

const Value::Array& Value::as_array() const {
  LLMRT_CHECK(type_ == Type::Array, std::string("json: expected array, got ") + type_name(type_));
  return array_;
}

const Value::Object& Value::as_object() const {
  LLMRT_CHECK(type_ == Type::Object, std::string("json: expected object, got ") + type_name(type_));
  return object_;
}

bool Value::contains(std::string_view key) const { return find(key) != nullptr; }

const Value* Value::find(std::string_view key) const {
  if (type_ != Type::Object) return nullptr;
  for (const auto& kv : object_) {
    if (kv.first == key) return &kv.second;
  }
  return nullptr;
}

const Value& Value::at(std::string_view key) const {
  const Value* v = find(key);
  LLMRT_CHECK(v != nullptr, "json: missing key '" + std::string(key) + "'");
  return *v;
}

int64_t Value::get_int(std::string_view key, int64_t fallback) const {
  const Value* v = find(key);
  if (v == nullptr || v->is_null()) return fallback;
  return v->as_int();
}

double Value::get_double(std::string_view key, double fallback) const {
  const Value* v = find(key);
  if (v == nullptr || v->is_null()) return fallback;
  return v->as_number();
}

bool Value::get_bool(std::string_view key, bool fallback) const {
  const Value* v = find(key);
  if (v == nullptr || v->is_null()) return fallback;
  return v->as_bool();
}

std::string Value::get_string(std::string_view key, std::string fallback) const {
  const Value* v = find(key);
  if (v == nullptr || v->is_null()) return fallback;
  return v->as_string();
}

const Value::Array* Value::get_array(std::string_view key) const {
  const Value* v = find(key);
  if (v == nullptr || v->is_null()) return nullptr;
  return &v->as_array();
}

size_t Value::size() const {
  if (type_ == Type::Array) return array_.size();
  if (type_ == Type::Object) return object_.size();
  return 0;
}

const Value& Value::operator[](size_t index) const {
  LLMRT_CHECK(type_ == Type::Array, "json: [] requires an array");
  LLMRT_CHECK(index < array_.size(), "json: array index out of range");
  return array_[index];
}

std::string Value::dump(int indent) const {
  std::string out;
  dump_impl(*this, out, indent, 0);
  return out;
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

Value parse(std::string_view text) { return Parser(text).parse_document(); }

Value parse_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  LLMRT_CHECK(in.good(), "json: cannot open '" + path + "'");
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  LLMRT_CHECK(!in.bad(), "json: failed reading '" + path + "'");
  return Parser(text).parse_document();
}

}  // namespace json
}  // namespace llmrt

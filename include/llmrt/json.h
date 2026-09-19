// Minimal, dependency-free JSON parser.
//
// Scope: exactly what the runtime needs to read model metadata --
//   * Qwen3 config.json / generation_config.json  (flat objects, numbers, bools)
//   * the safetensors header                       (nested arrays of ints)
//   * the golden-data manifest written by tools/gen_oracle.py
//
// Not supported on purpose: duplicate-key detection, streaming, JSON5.
// Object key order IS preserved, so tensor tables print deterministically.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llmrt {
namespace json {

enum class Type : uint8_t { Null, Bool, Number, String, Array, Object };

inline const char* type_name(Type t) {
  switch (t) {
    case Type::Null: return "null";
    case Type::Bool: return "bool";
    case Type::Number: return "number";
    case Type::String: return "string";
    case Type::Array: return "array";
    case Type::Object: return "object";
  }
  return "?";
}

class Value {
 public:
  using Array = std::vector<Value>;
  using Object = std::vector<std::pair<std::string, Value>>;

  Value() = default;

  static Value make_null();
  static Value make_bool(bool b);
  static Value make_int(int64_t i);
  static Value make_double(double d);
  static Value make_string(std::string s);
  static Value make_array(Array a);
  static Value make_object(Object o);

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_number() const { return type_ == Type::Number; }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_object() const { return type_ == Type::Object; }

  // True when the number was written without a fraction/exponent and fits int64.
  bool is_integer() const { return type_ == Type::Number && is_int_; }

  // Typed accessors. All throw llmrt::Error on a type mismatch.
  bool as_bool() const;
  int64_t as_int() const;
  double as_double() const;
  const std::string& as_string() const;
  const Array& as_array() const;
  const Object& as_object() const;

  // Number accessor that accepts int-or-double.
  double as_number() const;

  // ---- object lookup -----------------------------------------------------
  bool contains(std::string_view key) const;
  // Returns nullptr when absent or when *this is not an object.
  const Value* find(std::string_view key) const;
  // Throws when absent.
  const Value& at(std::string_view key) const;

  // Convenience getters with defaults; also accept a missing key silently.
  int64_t get_int(std::string_view key, int64_t fallback) const;
  double get_double(std::string_view key, double fallback) const;
  bool get_bool(std::string_view key, bool fallback) const;
  std::string get_string(std::string_view key, std::string fallback = "") const;
  // Throws if the key exists but is not an array.
  const Array* get_array(std::string_view key) const;

  // ---- array/object size and indexing -----------------------------------
  size_t size() const;
  const Value& operator[](size_t index) const;

  // Serialises back to JSON (used for debugging output).
  std::string dump(int indent = 0) const;

 private:
  Type type_ = Type::Null;
  bool bool_ = false;
  int64_t int_ = 0;
  double double_ = 0.0;
  bool is_int_ = false;
  std::string string_;
  Array array_;
  Object object_;
};

// Parses `text`. Throws llmrt::Error with line/column on malformed input.
Value parse(std::string_view text);

// Reads `path` fully and parses it.
Value parse_file(const std::string& path);

}  // namespace json
}  // namespace llmrt

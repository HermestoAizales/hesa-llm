#ifndef HESA_RESULT_HPP
#define HESA_RESULT_HPP

#include <expected>
#include <stdexcept>
#include <string>
#include <system_error>

namespace hesa {

enum class Error : int {
    SUCCESS = 0,
    OUT_OF_MEMORY,
    INVALID_ARGUMENT,
    BACKEND_NOT_AVAILABLE,
    MODEL_LOAD_FAILED,
    TENSOR_NOT_FOUND,
    SHAPE_MISMATCH,
    FILE_NOT_FOUND,
    INVALID_FORMAT,
    CUDA_ERROR,
    METAL_ERROR,
    TOKENIZER_ERROR,
    SAMPLING_ERROR,
    SESSION_ERROR,
    INTERNAL_ERROR
};

class ErrorCategory : public std::error_category {
public:
    const char* name() const noexcept override { return "hesa"; }
    std::string message(int ev) const noexcept override {
        switch (static_cast<Error>(ev)) {
            case Error::SUCCESS:                 return "Success";
            case Error::OUT_OF_MEMORY:           return "Out of memory";
            case Error::INVALID_ARGUMENT:        return "Invalid argument";
            case Error::BACKEND_NOT_AVAILABLE:   return "Backend not available";
            case Error::MODEL_LOAD_FAILED:       return "Model load failed";
            case Error::TENSOR_NOT_FOUND:        return "Tensor not found";
            case Error::SHAPE_MISMATCH:          return "Shape mismatch";
            case Error::FILE_NOT_FOUND:          return "File not found";
            case Error::INVALID_FORMAT:          return "Invalid format";
            case Error::CUDA_ERROR:              return "CUDA error";
            case Error::METAL_ERROR:             return "Metal error";
            case Error::TOKENIZER_ERROR:         return "Tokenizer error";
            case Error::SAMPLING_ERROR:          return "Sampling error";
            case Error::SESSION_ERROR:           return "Session error";
            case Error::INTERNAL_ERROR:          return "Internal error";
            default:                             return "Unknown error";
        }
    }
};

inline const ErrorCategory& error_category() {
    static ErrorCategory cat;
    return cat;
}

inline std::error_code make_error_code(Error e) {
    return {static_cast<int>(e), error_category()};
}

template<typename T>
using Result = std::expected<T, std::error_code>;

inline Result<void> ok() { return {}; }

// Generic error factory — works for Result<T> for any T
template<typename T>
inline Result<T> make_error(Error e) {
    return std::unexpected{make_error_code(e)};
}

inline Result<void> error(Error e) {
    return std::unexpected{make_error_code(e)};
}

inline Result<void> error(const std::system_error& e) {
    return std::unexpected{e.code()};
}

#define HESA_CHECK(expr) \
    do { \
        auto _res = (expr); \
        if (!_res) return std::unexpected{_res.error()}; \
        (void)_res; \
    } while(0)

#define HESA_CHECK_VALUE(expr) \
    [&]() -> decltype(auto) { \
        auto _res = (expr); \
        if (!_res) return std::unexpected{_res.error()}; \
        return std::forward<decltype(*_res)>(*_res); \
    }()

} // namespace hesa

namespace std {
template<>
struct is_error_code_enum<hesa::Error> : true_type {};
}

#endif // HESA_RESULT_HPP

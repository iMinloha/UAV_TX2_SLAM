#ifndef DEEPVISION_DEEPEXCEPTION_HPP
#define DEEPVISION_DEEPEXCEPTION_HPP

#include <exception>

namespace DeepVision {

    // 用于显示异常信息的类
    class DeepException : public std::exception {
    public:
        DeepException(const char* message) : message(message) {}
        const char* what() const noexcept override {
            return message;
        }
    private:
        const char* message;
    };

}

#endif

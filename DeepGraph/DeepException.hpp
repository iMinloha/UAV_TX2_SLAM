#ifndef DEEPVISION_DEEPEXCEPTION_HPP
#define DEEPVISION_DEEPEXCEPTION_HPP

#include <exception>

namespace DeepVision {

    // ������ʾ�쳣��Ϣ����
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

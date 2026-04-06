#include <hesa/hesa.h>
#include <iostream>

int main() {
    // Basic smoke test: verify version string is defined
    if (HESA_VERSION_MAJOR == 0 && HESA_VERSION_MINOR == 1 && HESA_VERSION_PATCH == 0) {
        std::cout << "test passed: hesa-llm v" HESA_VERSION_STRING << std::endl;
        return 0;
    }
    std::cerr << "test failed: unexpected version" << std::endl;
    return 1;
}

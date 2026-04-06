#include <hesa/hesa.h>
#include <iostream>

int main() {
    std::cout << "hesa-llm v"
              << HESA_VERSION_MAJOR << "."
              << HESA_VERSION_MINOR << "."
              << HESA_VERSION_PATCH << std::endl;
    return 0;
}

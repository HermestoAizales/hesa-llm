#include <iostream>
#include "hesa/version.hpp"

int main() {
    std::cout << "hesa-llm v"
              << hesa::VERSION_MAJOR << "."
              << hesa::VERSION_MINOR << "."
              << hesa::VERSION_PATCH
              << std::endl;
    return 0;
}

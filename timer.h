#ifndef TIMER
#define TIMER
#include <iostream>
struct Timer {
    size_t start;
    Timer() {
        reset();
    }
    void reset() {
        start = clock();
    }
    double elapsed() {
        return ((double)clock() - start) / CLOCKS_PER_SEC;
    }
    void printElapsed(char* label) {
        std::cout << label << ": " << elapsed() << "s\n";
    }
};
#endif
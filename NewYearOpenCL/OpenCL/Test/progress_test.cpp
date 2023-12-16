/* test program for the progressbar class
 * 
 * Author: Luigi Pertoldi
 * Created: 9 dic 2016
 *
 * Compile: c++ -I. -o test test.cc
 * Usage: ./test
 *
 */

#include <iostream>
#include <thread>
#include <chrono>

#include "../../ThirdParty/ProgressBar/progressbar.hpp"

#define sleep_time 50

int main() {

    int N = 10000;

    progressbar bar(N);
    bar.set_output_stream(std::cout);

    for ( int i = 0; i < N; i++ ) {
        bar.update();

        // the program...
        std::this_thread::sleep_for( std::chrono::microseconds(sleep_time) );
    }

    std::cout << std::endl;


    return 0;
}

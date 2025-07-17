#include <iostream>
#include <cassert>
#include <vector>

using namespace std;

// Placeholder mock recognizer function
int recognise_digit_mock(const vector<float>& pixels) {
    return 8; // Mock output for testing
}

int main() {
    vector<float> dummy_input(784, 0.0f);
    assert(recognise_digit_mock(dummy_input) == 8);
    cout << "✅ test_cpp.cpp passed.\n";
    return 0;
}

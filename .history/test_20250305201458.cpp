#include <iostream>
using namespace std;

enum Color {
    Red,
    Green,
    Blue,
    Yellow
};

int main() {
    Color c = Blue;
    cout << "Color value: " << c << endl;
    return 0;
}
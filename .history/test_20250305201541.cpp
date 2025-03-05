#include <iostream>
using namespace std;

enum Color {
    Red,
    Green,
    Blue,
    Yellow
} c ;

int main() {
    c = Blue;
    cout << "Color value: " << c << endl;
    return 0;
}
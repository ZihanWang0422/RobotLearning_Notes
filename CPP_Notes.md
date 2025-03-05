## Chapter 1 基本语法

1、结构：

- **对象 -** 对象具有状态和行为。例如：一只狗的状态 - 颜色、名称、品种，行为 - 摇动、叫唤、吃。对象是类的实例。
- **类 -** 类可以定义为描述对象行为/状态的模板/蓝图。
- **方法 -** 从基本上说，一个方法表示一种行为。一个类可以包含多个方法。可以在方法中写入逻辑、操作数据以及执行所有的动作。

```c++
#include <iostream> //头文件
using namespace std; //告诉编译器使用 std 命名空间。
 
// main() 是程序开始执行的地方
 
int main()
{
   cout << "Hello World"; // 输出 Hello World
   return 0;  //终止 main( )函数，并向调用进程返回值 0
}
//注意分号
```

2、关键字

[C++ 的关键字（保留字）完整介绍 | 菜鸟教程](https://www.runoob.com/w3cnote/cpp-keyword-intro.html)

3、注释

//单行 

/* ... */ 多行注释



## Chapter 2 数据类型

1、基本类型：（1字节 = 8位）

| 类型          | 关键字  |
| ------------- | ------- |
| 布尔型（1）   | bool    |
| 字符型（1）   | char    |
| 整型（4）     | int     |
| 浮点型（4）   | float   |
| 双浮点型（8） | double  |
| 无类型        | void    |
| 宽字符型      | wchar_t |

* typedef short int wchar_t;

  ​

2、修饰符：

| 修饰符     | 描述                                   | 示例                   |
| ---------- | -------------------------------------- | ---------------------- |
| `signed`   | 表示有符号类型（默认）                 | `signed int x = -10;`  |
| `unsigned` | 表示无符号类型                         | `unsigned int y = 10;` |
| `short`    | 表示短整型                             | `short int z = 100;`   |
| `long`     | 表示长整型                             | `long int a = 100000;` |
| `const`    | 表示常量，值不可修改                   | `const int b = 5;`     |
| `volatile` | 表示变量可能被意外修改，禁止编译器优化 | `volatile int c = 10;` |
| `mutable`  | 表示类成员可以在 `const` 对象中修改    | `mutable int counter;` |



3、别名：

* typedef  `typedef int MyInt` -> `MyInt distance`

* using `using MyInt = int`

  ​

4、枚举类型：

```    c++
enum 枚举名{ 
     标识符[=整型常数], 
     标识符[=整型常数], 
... 
    标识符[=整型常数]
} 枚举变量;

eg：
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
```



5、类型转换：

（1）静态转换（不进行类型检查）：

强制转换，用于比较相似类型的转换。

```c++
int i = 10;
float f = static_cast<float>(i); // 静态将int类型转换为float类型
```

（2）动态转换(进行类型检查)：

继承结构中向下转换，用于将一个基类指针或引用转换为派生类指针或引用。



（3）常量转换：

用于将const类型转换为非const类型

```c++
const int i = 10;
int& r = const_cast<int&>(i); // 常量转换，将const int转换为int
```

（4）重新解释转换（不进行类型检查）：

将一个数据类型的值重新解释为另一个数据类型的值，通常用于在不同的数据类型之间进行转换。

```c++
const int i = 10;
int& r = const_cast<int&>(i); // 常量转换，将const int转换为int
```



## Chapter 3 变量

1、定义：

```c++
type variable_list;
type variable_name = value;
```

2、变量声明：

注意 - 可以多次声明一个变量，但变量只能在某个文件、函数或代码块中被定义一次

```c++
//变量在头部就已经被声明，但它们是在主函数内被定义和初始化的
#include <iostream>
using namespace std;
 
// 变量声明
extern int a, b;
extern int c;
extern float f;
  
int main ()
{
  // 变量定义
  int a, b;
  int c;
  float f;
 
  // 实际初始化
  a = 10;
  b = 20;
  c = a + b;
 
  cout << c << endl ;
 
  f = 70.0/3.0;
  cout << f << endl ;
 
  return 0;
}
```














































































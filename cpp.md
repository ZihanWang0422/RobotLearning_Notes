

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

  

### 2、修饰符：

| 修饰符      | 描述                   | 示例                   |
| ----------- | ---------------------- | ---------------------- |
| `signed`    | 表示有符号类型（默认） | `signed int x = -10;`  |
| `unsigned`  | 表示无符号类型         | `unsigned int y = 10;` |
| `short`     | 表示短整型             | `short (int) z = 100;` |
| `long`      | 表示长整型             | `long int a = 100000;` |
| `long long` | 范围比long更大         |                        |

* **signed、unsigned、long 和 short** 可应用于整型，**signed** 和 **unsigned** 可应用于字符型，**long** 可应用于双精度型。
* 修饰符 **signed** 和 **unsigned** 也可以作为 **long** 或 **short** 修饰符的前缀





3、别名：

* typedef  `typedef int MyInt` -> `MyInt distance`

* using `using MyInt = int`

  

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



3、变量作用域：

- 在函数或一个代码块内部声明的变量，称为**局部变量**。
- 在函数参数的定义中声明的变量，称为**形式参数**。
- 在所有函数外部声明的变量，称为**全局变量**。

```c++
#include <iostream>

int main() {
    int a = 10;
    {
        int a = 20;  // 块作用域变量
        std::cout << "块变量: " << a << std::endl;
    }
    std::cout << "外部变量: " << a << std::endl;
    return 0;
}
```



4、初始化变量：

| 数据类型 | 初始化默认值 |
| -------- | ------------ |
| int      | 0            |
| char     | '\0'         |
| float    | 0            |
| double   | 0            |
| pointer  | NULL         |



### 5、变量类型限定符：

| 限定符   | 含义                                                         |
| -------- | ------------------------------------------------------------ |
| static   | 用于定义静态变量，表示该变量的作用域**仅限于当前文件或当前函数内**，不会被其他文件或函数访问。<br />const 默认为static，且不可被修改 |
| extern   | 用于声明具有外部链接的变量或函数，它们可以在**多个文件**之间共享。默认情况下，**全局变量和函数具有 extern 存储类**。 <br >在一个文件中使用extern声明另一个文件中定义的全局变量或函数，可以实现跨文件共享。 |
| mutable  | mutable 用于修饰类的成员变量。被 mutable 修饰的成员变量可以被修改，**即使它们所在的对象是 const 的。** |
| volatile | 修饰符 **volatile** 告诉该变量的值可能会被程序以外的因素改变，如硬件或其他线程。 |
| restrict | 由 **restrict** 修饰的指针是唯一一种访问它所指向的对象的方式。只有 C99 增加了新的类型限定符 restrict。 |

```C++
#include <iostream>

// 全局变量，多个文件共享，默认存储类为extern
int globalVar;

//全局变量，作用域限制在本文件内
static int count = 10;

void function() {
    // 局部静态变量，具有静态存储期，生命周期贯穿整个程序
    static int staticVar = 20;

    const int constVar = 30; // const变量默认具有static存储期

    // 尝试修改const变量，编译错误
    // constVar = 40;

    // mutable成员变量，可以在const成员函数中修改
    class MyClass {
    public:
        mutable int mutableVar;

        void constMemberFunc() const {
            mutableVar = 50; // 允许修改mutable成员变量
        }
    };

    // 线程局部变量，每个线程有自己的独立副本
    thread_local int threadVar = 60;
}

int main() {
    extern int externalVar; // 声明具有外部链接的变量

    function();

    return 0;
}
```



第一个文件：main.cpp

```c++
#include <iostream>
 
int count ;
extern void write_extern();
 
int main()
{
   count = 5;
   write_extern();
}
```



第二个文件：support.cpp

```c++
//第二个文件中的 extern 关键字用于声明已经在第一个文件 main.cpp 中定义的 count
#include <iostream>
 
extern int count;
 
void write_extern(void)
{
   std::cout << "Count is " << count << std::endl;
}
```





## Chapter 4 常量

1、整数：

（1）前缀

0X(x) —— 十六进制

0 ——八进制

不带前缀 ——十进制

（2）后缀

U —— 无符号整数

L ——长整数



2、浮点数：

（1）小数表示

必须包含整数、小数或同时包含两者

（2）指数表示

e/E后面的数字表示10的幂次



3、布尔常量：

true — 真

false — 假



4、字符常量：

单引号内 — 'x'

宽字符常量 — L'x' 此时必须存储在 **wchar_t** 内

| 转义序列   | 含义                       |
| ---------- | -------------------------- |
| \\         | \ 字符                     |
| \'         | ' 字符                     |
| \"         | " 字符                     |
| \?         | ? 字符                     |
| \a         | 警报铃声                   |
| \b         | 退格键                     |
| \f         | 换页符                     |
| \n         | 换行符                     |
| \r         | 回车                       |
| \t         | 水平制表符                 |
| \v         | 垂直制表符                 |
| \ooo       | 一到三位的八进制数         |
| \xhh . . . | 一个或多个数字的十六进制数 |



5、字符串常量：

双引号内

可以使用`\`做分隔符，对长字符串进行分行



6、定义常量：

**规范：常量字母全大写**

（1）#define





```c++
#define identifier value
//注意没有分号
#define LENGTH 10
```



（2）const

```c++
const type variable = value;
//注意有分号
const int LENGTH 10
```



## Chapter 5 运算符

1、算术运算符

| 运算符 | 描述                                                         |
| ------ | ------------------------------------------------------------ |
| +      | 把两个操作数相加                                             |
| -      | 从第一个操作数中减去第二个操作数                             |
| *      | 把两个操作数相乘                                             |
| /      | 分子除以分母                                                 |
| %      | 取模运算符，整除后的余数                                     |
| ++     | [自增运算符](https://www.runoob.com/cplusplus/cpp-increment-decrement-operators.html)，整数值增加 1 |
| --     | [自减运算符](https://www.runoob.com/cplusplus/cpp-increment-decrement-operators.html)，整数值减少 1 |

* 前缀：`x++`在表达式计算之后完成自增或自减
* 后缀：`++x`在表达式计算之前完成自增或自减



2、关系运算符

| 运算符 | 描述                                                         | 实例              |
| ------ | ------------------------------------------------------------ | ----------------- |
| ==     | 检查两个操作数的值是否相等，如果相等则条件为真。             | (A == B) 不为真。 |
| !=     | 检查两个操作数的值是否相等，如果不相等则条件为真。           | (A != B) 为真。   |
| >      | 检查左操作数的值是否大于右操作数的值，如果是则条件为真。     | (A > B) 不为真。  |
| <      | 检查左操作数的值是否小于右操作数的值，如果是则条件为真。     | (A < B) 为真。    |
| >=     | 检查左操作数的值是否大于或等于右操作数的值，如果是则条件为真。 | (A >= B) 不为真。 |
| <=     | 检查左操作数的值是否小于或等于右操作数的值，如果是则条件为真。 | (A <= B) 为真     |

3、逻辑运算符

| 运算符 | 描述                                                         | 实例                 |
| ------ | ------------------------------------------------------------ | -------------------- |
| &&     | 称为逻辑与运算符。如果两个操作数都 true，则条件为 true。     | (A && B) 为 false。  |
| \|\|   | 称为逻辑或运算符。如果两个操作数中有任意一个 true，则条件为 true。 | (A \|\| B) 为 true。 |
| !      | 称为逻辑非运算符。用来逆转操作数的逻辑状态，如果条件为 true 则逻辑非运算符将使其为 false。 | !(A && B) 为 true。  |

4、位运算符

| 运算符 | 描述                                                         | 实例(A = 0011 1100, B = 0000 1101)                           |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| &      | 按位与操作，按二进制位进行"与"运算。运算规则：`0&0=0;   0&1=0;    1&0=0;     1&1=1;` | (A & B) 将得到 12，即为 0000 1100                            |
| \|     | 按位或运算符，按二进制位进行"或"运算。运算规则：`0|0=0;   0|1=1;   1|0=1;    1|1=1;` | (A \| B) 将得到 61，即为 0011 1101                           |
| ^      | 异或运算符，按二进制位进行"异或"运算。运算规则：`0^0=0;   0^1=1;   1^0=1;  1^1=0;` | (A ^ B) 将得到 49，即为 0011 0001                            |
| ~      | 取反运算符，按二进制位进行"取反"运算。运算规则：`~1=-2;   ~0=-1;` | (~A ) 将得到 -61，即为 1100 0011，一个有符号二进制数的补码形式。 |
| <<     | 二进制左移运算符。将一个运算对象的各二进制位全部左移若干位（左边的二进制位丢弃，右边补0）。 | A << 2 将得到 240，即为 1111 0000                            |
| >>     | 二进制右移运算符。将一个数的各二进制位全部右移若干位，正数左补0，负数左补1，右边丢弃。 | A >> 2 将得到 15，即为 0000 1111                             |



5、赋值运算符

| 运算符 | 描述                                                         | 实例                            |
| ------ | ------------------------------------------------------------ | ------------------------------- |
| =      | 简单的赋值运算符，把右边操作数的值赋给左边操作数             | C = A + B 将把 A + B 的值赋给 C |
| +=     | 加且赋值运算符，把右边操作数加上左边操作数的结果赋值给左边操作数 | C += A 相当于 C = C + A         |
| -=     | 减且赋值运算符，把左边操作数减去右边操作数的结果赋值给左边操作数 | C -= A 相当于 C = C - A         |
| *=     | 乘且赋值运算符，把右边操作数乘以左边操作数的结果赋值给左边操作数 | C *= A 相当于 C = C * A         |
| /=     | 除且赋值运算符，把左边操作数除以右边操作数的结果赋值给左边操作数 | C /= A 相当于 C = C / A         |
| %=     | 求模且赋值运算符，求两个操作数的模赋值给左边操作数           | C %= A 相当于 C = C % A         |
| <<=    | 左移且赋值运算符                                             | C <<= 2 等同于 C = C << 2       |
| >>=    | 右移且赋值运算符                                             | C >>= 2 等同于 C = C >> 2       |
| &=     | 按位与且赋值运算符                                           | C &= 2 等同于 C = C & 2         |
| ^=     | 按位异或且赋值运算符                                         | C ^= 2 等同于 C = C ^ 2         |
| \|=    | 按位或且赋值运算符                                           | C \|= 2 等同于 C = C \| 2       |



6、其他

| 运算符               | 描述                                                         |
| -------------------- | ------------------------------------------------------------ |
| sizeof               | [sizeof 运算符](https://www.runoob.com/cplusplus/cpp-sizeof-operator.html)返回变量的字节大小。例如，sizeof(a) 将返回 4，其中 a 是整数。 |
| Condition ? X : Y    | [条件运算符](https://www.runoob.com/cplusplus/cpp-conditional-operator.html)。如果 Condition 为真 ? 则值为 X : 否则值为 Y。 |
| ,                    | [逗号运算符](https://www.runoob.com/cplusplus/cpp-comma-operator.html)会顺序执行一系列运算。整个逗号表达式的值是以逗号分隔的列表中的最后一个表达式的值。 |
| .（点）和 ->（箭头） | [成员运算符](https://www.runoob.com/cplusplus/cpp-member-operators.html)用于引用类、结构和共用体的成员。 |
| Cast                 | [强制转换运算符](https://www.runoob.com/cplusplus/cpp-casting-operators.html)把一种数据类型转换为另一种数据类型。例如，int(2.2000) 将返回 2。 |
| &                    | [指针运算符 &](https://www.runoob.com/cplusplus/cpp-pointer-operators.html) 返回变量的地址。例如 &a; 将给出变量的实际地址。 |
| *                    | [指针运算符 *](https://www.runoob.com/cplusplus/cpp-pointer-operators.html) 指向一个变量。例如，*var; 将指向变量 var。 |



## Chapter 6 循环

1. while

```c++
//当条件为真时执行循环
while(condition)
{
   statement(s);
}
```



2. for循环

* **init** 首先执行，且只执行一次；此处也可以只有一个分号。
* **condition**  判断为真，则执行循环主体，如果为假，则不执行循环主体，跳转for的下一条语句
* 执行完for循环主体后，跳转回**increment**，该条件可以为空，留分号
* 再次判断条件

```c++
for ( init; condition; increment )
{
   statement(s);
}
```

🤔空for循环可以实现无限循环

```c++
#include <iostream>
using namespace std;
 
int main ()
{
 
   for( ; ; )
   {
      printf("This loop will run forever.\n");
   }
 
   return 0;
}
```



3. do... while

至少执行一次statement

```c++
do
{
   statement(s);

}while( condition );
```



4. 嵌套循环

```c++
#include <iostream>
using namespace std;
 
int main ()
{
    int i, j;
    for(i=2; i<100; i++) {
        for(j=2; j <= (i/j); j++) {
            if(!(i%j)) {
                break; // 如果找到，则不是质数
            }
        }
        if(j > (i/j)) {
            cout << i << " 是质数\n";
        }
    }
    return 0;
}
```



5. 循环控制

（1）break: 当 **break** 语句出现在一个循环内时，循环会立即终止，且程序流将继续执行紧接着循环的下一条语句。

<img src="./cpp.assets/c-break-statement-works.jpg" alt="img" style="zoom: 80%;" />

（2）continue: 跳过当前循环中的代码，强迫开始下一次循环

<img src="./cpp.assets/c-continue-statement-works.jpg" alt="img" style="zoom:80%;" />

（3）goto:



## Chapter 7 判断

1. if

任何**非零**和**非空**的值假定为 **true**，把**零**或 **null** 假定为 **false**

```c++
if(boolean_expression)
{
   // 如果布尔表达式为真将执行的语句
}
```



2. if.. else..

```c++
if(boolean_expression)
{
   // 如果布尔表达式为真将执行的语句
}
else
{
   // 如果布尔表达式为假将执行的语句
}
```

```C++
if(boolean_expression 1)
{
   // 当布尔表达式 1 为真时执行
}
else if( boolean_expression 2)
{
   // 当布尔表达式 2 为真时执行
}
else if( boolean_expression 3)
{
   // 当布尔表达式 3 为真时执行
}
else 
{
   // 当上面条件都不为真时执行
}

```



3. 嵌套if

```c++
if( boolean_expression 1)
{
   // 当布尔表达式 1 为 true 时执行
   if(boolean_expression 2)
   {
      // 当布尔表达式 2 为 ture 时执行
   }
}

```

```c++
if (condition1) {
   // 如果 condition1 为 true，则执行此处的代码块
   if (condition2) {
      // 如果 condition2 也为 true，则执行此处的代码块
   }
   else {
      // 如果 condition2 为 false，则执行此处的代码块
   }
}
else {
   // 如果 condition1 为 false，则执行此处的代码块
}
```



4. switch

```c++
switch(expression){
    case constant-expression  :
       statement(s);
       break; // 可选的
    case constant-expression  :
       statement(s);
       break; // 可选的
  
    // 您可以有任意数量的 case 语句
    default : // 可选的
       statement(s);
}
```






















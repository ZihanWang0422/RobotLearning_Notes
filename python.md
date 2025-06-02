
## 第一章 输入输出

### 一、print函数的使用

1、直接引用：
`print("Hello world")`

2、输出变量：
`print(a)`

3、末尾自带换行：
如要实现不换行，在变量末尾加上`end=""`
`print(x,end=" ")`

4、多个字符串可以放一行中用逗号隔开，逗号后面自动输出空格

```python
print('num', 'list', 'tuple')
>>>num list tuple
```

5、理解python中的函数与变量：
函数：expression()
然后可对变量赋值函数：a = expression(x)

```python
print(a)
or
print(expression(x))
```

### 二、行与缩进

1、缩进：同一个代码块的语句必须含有相同的缩进空格 (<font color=red>一定注意缩进的空格 </font>)
2、多行语句：

```python
total = item_one + \
        item_two + \
        item_three 
```

在{},[],()中的多行语句无需使用`\`

```python
total = ['item_one', 'item_two', 'item_three',
        'item_four', 'item_five']
```

同一行可以使用多条语句，中间用分号隔开

### 三、等待用户输入

```python
a = input('描述性语言'所需输入的数 ) #然后用户在终端输入数字，即可对a完成赋 (相当于使一种交互式操作)
```

### 四、导入模式  命令行参 (存疑)

### 五、格式化输出

1、%法:

%d int
%f float
%s string  #万能，可以将任何数据类型转换成字符串
%x 十六进制整数
%2d 数字占两格
%.2f 小数点后保留两位

格式化操作符：

`.` 定义宽度或小数点精度

`-` 用作左对齐

`+` 在正数前显示正号

`space` 在正数前显示空格

`0` 显示的数字前填充0

`m.n.` m显示的最小总宽度，n小数点后的位数

PS： %后面要有空格

```python
print('I am %d %s old' % (20, 'years' ))  #多个格式化的要加括号
```

2、format:

```python
num = 12
name = 'James'
print('My age is {} and my name is {}'.format(num,name))
print(f"My age is {num} and my name is {name}")
```



## 第二章  数据类型

type()可以查询所指对象的类型

isinstance(object,classinfo)判断一个函数是否是一个已知的类型
object:实例对象
classinfo:直接或间接类名、基本类型或者由它们组成的元素
相匹 ->return True; else return False

### 一、变量的命名

1、字母或下划线打头，不能数字打头
2、不能有空格
3、保留词

```python
'False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'
```

4、对多个变量赋值
a=b=c=1
a, b, c = 1, 2, "coc"(分别赋值 )

### 二、字符串

1、引号可以是`'`或`"`，且可互相嵌套
`'''`or`"""`可以指定多行字符串

```python
paragraph = """这是一个段 
由多行句子组 """
```

转义字符：`\'` `\"` 引号
`%%` 百分符
`\\` 反斜杠号
`\b` 退符
`\000`  
`\v` 纵向制表符
`\t` 横向制表符
`\r` 回车，将 \r 后面的内容移到字符串开头，并逐一替换开头部分的字符，直至将 \r 后面的内容完全替换完为止

```python
print('Hello\rWorld!')
>>>World!
```

`\f` 换页

```python
print('Hello \f World!')
>>>
```

2、修改大小写

```python
name->变量
   name.title() 将单词的首字母改成大写，中间多个单词组成的字符串，会讲每个单词的首字母均大写
   name.upper() 将每个小写字母改成大 
   name.lower() 将每个大写字母改成小 
print(name.fun())->即可输出相应结果
```

3、合并字符串

```python
变量合成：A = "zifuchuan" + "zifuchuan" 
字符串合成：print("zifuchuan" + "zifuchuan")
```

4、重复字符串

```python
print(str*2)
>>将str的内容输出两 
```

5、制表符 (相当于给内容前面空四格)

```python
print("\tpython")
   python
```

6、换行符 (同理于c中的\n)

```python
print("python\njava\n)
python
java
```

相结合：

```python
\n\tzifuchuan
```

在字符串前面使用`r`可以让反斜杠不发生转义

```python
print(r"hello\n")
>>>hello\n  #相当于使\n直接输出
```

7、删除空白：
暂时删除：删除右侧的空白->rstrip();删除左侧的空 ->lstrip();两侧均删 ->strip()
永久删除：将rstrip后的值赋予到变量中去

```python
language=' python '
print(language.lstrip())
print(language.rstrip())
print(language.strip())

python
 python
python
```

8、字符串的索引：(字符串不可更 )
从左往右以0开始，从右往左以-1开 
左闭右开

```python
str = "abcdefg"
print(str[0:-1])  #输出从第一个到倒数第二个的所有字符abcdef
print(str[2:5])   #输出从第三个开始到第五个的字符cde **注意，末尾索引的字符不算在内
print(str[2])     #输出从第三个开始后的所有字 
print(str[1:5:2]) #输出从第二个开始到第五个且步长 2(中间隔一个字符）的字符bd
print(str[:]) print([0:]) print([:7]) #输出整个字符 
print(str[:x]) #输出序列号为x的字符前的字 
print(str[x:]) #输出序列号为x的字符后面的字符
```

### 三、数 

数字类型 
int(整数)：python中没有long 
bool(布尔)：True,False(是int的子 )
float(浮点 )：a + bj or complex(a,b)
complex(复数)

1、打印数字字符串函数str():

```python
age = 23
message = " happy " + str(age) + " birthday "
print(message)

 happy 23 birthday 

```

2、运算：
```python
除法：`/`->float
     `//`->int
乘方：`**`
取余：`%`
混合计算时，整型会转换成浮点 
```

### 四、列表(list)

1、构造列表：
list = ['a', 'b', 'c']
list = [1, 2, 3]
list = [] #空列 
从左往右以0开始，从右往左以-1开始，均类似于字符串索 

2、改变元素：
改变单个元素：list[num]=...
改变一串元素：list[1:3]=[x,x]
置零对应的元素值：

```python
list=[9,2,13,14,15,6]
list[2:5] = []
>>>list=[9,2,6](置零位次后，后面的位次自动补 )
```

3、二维列表：

```python
p = ['asp', 'php']
s = ['python', 'java', p, 'scheme']  #得到'php's[2][1]
```

4、创建指定长度的空列表：

```python
#现在就实现了创建长度为10的空列表
len = 10
a = [NONE]*len

#后面可以对空列表进行赋值
a[0: 9] = list[0: 9]
```



### 五、元组 (tuple)

1、构造元组：
tuple = ('a', 'b', 'c')
tuple = (1, 2, 3)
元组的元素不能修改
其他均与列表类似

2、构造特殊元组：

```python
tup1 = ()    #空元 
tup2 = (20,) #一个元素，需要在元素后加逗号。否则就输出一个数
```

3、二维元组：
可在元组内嵌套一个列表，从而通过改变列表的元素而改变元组的元素

### 六、集合 (set)

* 表示无序不重复的元素序列->给数据去 
* 终端输出格式 
{'a','c','d'}可能会乱 
* 集合中的元素必须是不可变 

1、构造集合：
使用{} or set() 来创建集 
创建空集合使  set() 不能用{}->创建空字 

```python
sites = {val1,val2,...}
or
sites=set(val) #但注意此时只能传入一个参 ->字符串，list，tuple 
```

2、输出集合：
print(sites)
则重复的元素被自动删 

3、成员测试：

```python
if val1 in sites :
  prnt(...)
else :
  print(...)
```

4、使用set进行集合的运算：

```python
a-b  #差集：输出a中有的b中没有的元素
a|b  #并集：输出a和b的并 
a&b  #交集：输出a和b的交 
a^b  #补集：输出a和b中不同时存在的元 
```

### 七、字典(dict)

是一种映射类型：由键(key): (value)组成
注意：键必须为不可变类型(数字,字符,元组)

1、索引：

```python
dict = {}             #创建空字 
dict = {'name': 'Wang','age':20}

print(dict)           #输出完整字典
print(dict.keys())    #输出所有的键值
print(dict.values())  #输出所有的值
print(dict['name'])   #输出键为'name'的 
```

2、构造字典：

*  键值对序列 

 ```python
a=dict([('name',1), ('age',2),('site',3)])  #注意中括号不要漏 
print(a)
{'name': 1, 'age': 2, 'site': 3}
 ```

* 关键字参数：

```python
a=dict(name=1,age=2,site=3)
print(a)
>>>{ 'name': 1, 'age': 2, 'site':3 }

empty = dict()
print(empty)
>>>{}  #创建空集 
```

* 创建可迭代对象：

```python
num = dict(dict(zip(['x', 'y', 'z'], [1, 2, 3])))
print(num)
>>>{'x': 1, 'y': 2, 'z': 3}
```

* 映射类型 

```python
num = dict({'x':4, 'y': 5})
print(num)
>>>{'x': 4, 'y': 5}

num = {'x': 4, 'y': 5}
print(num)
>>>{'x': 4, 'y': 5}
```

* 推导式赋值法 

 ```python
a={x: x**2 for x in (2,4,6)}
print(a)
{2: 4, 4: 16, 6: 36}
 ```

### 八、推导式

#### 1、列表 

[表达  for 变量 in 列表]
[out_exp_res for out_exp in input_list]

或 

[表达  for 变量 in 列表 if 条件]
[out_exp_res for out_exp in input_list if condition]

* out_exp_res：列表生成元素表达式，可以是有返回值的函数 
* for out_exp in input_list：迭  input_list   out_exp 传入  out_exp_res 表达式中 
* if condition：条件语句，可以过滤列表中不符合条件的值 

过滤掉长度小于或等于3的字符串列表，并将剩下的转换成大写字母：

```python
names = ['Bob','Tom','alice','Jerry','Wendy','Smith']
new_names = [name.upper()for name in names if len(name)>3]
print(new_names)
>>>['ALICE', 'JERRY', 'WENDY', 'SMITH']
```

计算 30 以内可以  3 整除的整数：

```python
multiples = [i for i in range(30) if i % 3 == 0]
print(multiples)
>>>[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
```

#### 2、字典

{ key_expr: value_expr for value in collection }

 

{ key_expr: value_expr for value in collection if condition }

* key_expr: 字典的key 
* value_exper: value的 ,是关于key的表达式或函 
* value: collection中的key 

以字符串及其长度创建字典 

```python
dict = ['Google','Runoob', 'Taobao']
new_dict = {key:len(key) for key in dict}
print(new_dict)
>>>{'Google': 6, 'Runoob': 6, 'Taobao': 6}
```

#### 3、集合 

{ expression for item in Sequence }

{ expression for item in Sequence if conditional }

* expression: 关于item的表达式或函 
* item: 在Sequence中的 
* Sequence: 数据 (集合、元组、字符串 ),最后结果转化为集合
  

计算数字1,2,3的平方数:

```python
setnew = {i**2 for i in (1,2,3)}
print(setnew)
>>>{1,4,9}
```

判断不是 abc 的字母并输出 

```python
a = {x for x in 'abracadabra' if x not in 'abc'}  #注意not的用 
print(a)
>>>{'d', 'r'}
```

#### 4、元组

(expression for item in Sequence )

(expression for item in Sequence if conditional )

生成一个包含数 1-9的元组：

```python
a = (x for x in range(1,10))
print(a)
>>><generator object <genexpr> at 0x000001C3D1AD0548>  #返回生成器对 
print(tuple(a))
>>>(1, 2, 3, 4, 5, 6, 7, 8, 9)  #tuple()将list、range转换为元 
```

### 九、数据类型转换

1、隐式类型转换：
较低的数据类 (int)会自动转换成较高的数据类 (float)

```python
num_int=12
num_float=1.3
num_new=num_int + num_float
print(num_new)
>>>13.3
```

2、显式类型转换：

#### map(fun,iterable)

根据函数对一个或多个序列进行指定映射(返回值是一个新的对象，旧对象不做出修改)

```python
#与lambda函数结合
def mapTest():
    a = [1, 2, 3, 4, 5, 6]
    r = map(lambda x:x**2, a)
    print(list(r))
#lambda处理多个对象
def mapTest():
    a = [1, 2, 3, 4, 5, 6]
    b = [9, 8, 7, 6, 5, 4]
    r = map(lambda x,y:(x+y,x*y),a,b)
    print(list(r))
```

#### filter(fun,iterable)

筛选器：将最后返回值为True的元素(符合fun的值)到新列表中

```python
#过滤掉为0的数
list_num = [1, 3, 0, 2, 0 ]
print(list(filter(lambda x: x, list_num)))

#过滤大小写
list_word = ['a', 'B', 'c', 'd', 'E']
print(list(filter(lambda x: x.isupper(), list_word)))
print(list(filter(lambda x: x.islower(), list_word)))

```



#### int(x)

强制转换成整数(x为纯数字，且不能有base)

```python
int(3.4)
>>>3
```

```python
  int(str, base)
  视str为base类型，并将其转换成十进制数，base为十进制时，可以省略
  注意：str须是整数
```

```python
int('1001', 2)
>>>10
```

#### float()

强制转换成浮点型

#### str() 强制转换成字符串

#### complex([real[, imag]])

real-int,float,字符串
imag-int,float

```python
complex(1, 2)
complex(1)
complex('1')
complex('1+2j')  #注意+两遍不能有空 
>>>(1+2j)
(1+0j)
(1+0j)
(1+2j)
```

#### repr()  

将读取到的格式字符转化成相应的转义字 ,返回string格式

```python
s="物品\t单价\t数量\n包子\t1\t2"
print(s)
print(repr(s))
>>>
物品 单价 数量
包子 1     2
```

#### eval(expression[, globals[, locals]])  

可以将字符串的引号去掉，保留字符的原本特征

expression -- 表达式
globals -- 变量作用域，全局命名空间，如果被提供，则必须是一个字典对象
locals -- 变量作用域，局部命名空间，如果被提供，可以是任何映射对象

将文本中读取的字符串形式的列表转换成列表

```python
zifu=" ['1709020230', '1707030416', '0', '0', '0']  "
print(type(zifu))
ls =eval(zifu)
print(type(ls))
>>>
<class 'str'>
<class 'list'>
```

#### tuple()

将字符串、列表、字典、集合转换为元组

```python
a = 'www'
print(tuple(a))
>>>('w', 'w', 'w')

a={'www':123,'aaa':234}
print(tuple(a))
>>>('www', 'aaa')  #字典转换成元组只输出key 

a=set('abcd')
print(tuple(a))
>>>('a', 'd', 'b', 'c')
```

#### list() 

将字符串、元组、集合、字典转换成列表

#### frozenset([item]) 

返回一个冻结的集合，不能添加或删除任何元素(item-可迭代对 (列表、字典、元组等))

```python
a = frozenset(range(10))
print(a)
>>>frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})

b = frozenset('baidu')
print(b)
>>>frozenset({'b', 'i', 'a', 'd', 'u'})
```

解释：有的集合中的元素是另一个集合的情况，但普通集 (set),本身是可变的，则它的实例不能放在另一个集合中
故frozen可作为不可变集合，满足了作为集合中的元素的要 

#### chr(i)

返回当前整数对应的ASCII字符(i-10 16进制)

#### ord(i)

返回字符对应的整数

i-长度 1的字符串

#### hex(x)

10进制数转换成16进制数，结果以字符串形式表示

#### oct(x)

10进制数转换成8进制数，结果以字符串形式表示

### 十、运算符

#### 1、算术运算符

* 除法：`/`->float
     `//`->整数结果，但不一定是int类型
     PS 7//2 >>>3
    7.0//2 >>>3.0
    7//2.0 >>>3.0

* 乘方：`**`
* 取余：`%`
* 混合计算时，整型会转换成浮点型

#### 2、比较运算符(注意：返回值为bool类型True或False)

```python
==
!=
>
<
>=
<=
```

#### 3、赋值运算符

c xx=a <=> c = c xx a

```python
=
+= # a+=1 对a进行累加
-=
*=
/=
%=
**=
//=
:=  #海象运算符：可在表达式内部将变量赋 
```

```python
if (n := len(a)) > 10:
    print(f"List is too long ({n} elements, expected <= 10)")
```

#### 4、位运算 

* 按位与 &： 同为1 ，否则为0

* 按位或 |：  1 1

* 按位异或  ^：  相异 1

* 按位取反  ~：  对数据的每个二进制位取反,即把1变为0, 0变为1。~x 类似  -x-1

* 左移运算符： <<  运算数的各二进位全部左移若干位， "<<"右边的数指定移动的位数，高位丢弃，低位补0 

* 右移运算符：>>   ">>"左边的运算数的各二进位全部右移若干位 ">>"右边的数指定移动的位 

#### 5、逻辑运算 

PS 
字符串：有内容为true，空字符串为false

数字:   0为false，其他为true

bool:  True，False

 x exp y:

* bool与： and 若x为false，返回x的值，否则返回y的计算(同真为真，否则为假)

* bool或： or 若x为true，返回x的值，否则返回y的计算(同假为假，否则为真)

* bool非： not 若x为true，返回false；若x为false，返回true

#### 6、成员运算符

* in 在指定序列中找到值，返回True，否则返回False

* not in 在指定序列中没有找到值，返回True，否则返回False

#### 7、身份运算符

* is 判断两个标识符是不是引用自同一对象 <=> id(x) == id(y) 是一个对象返回True，否则返回False

* is not 判断两个标识符是不是引用自不同对  <=> id(x) != id(y) 不是同一对象返回True，否则返回False

PS: id() 读取对象的地址
is 判断两个变量的引用对象是否是同一个，==用于判断引用变量的值是否相 

```python
a = [1,2,3]
b = a
b is a True
b == a True

b = a[:]  #a[:]是深复制，a是浅复制，赋值a的话是赋值了指针，赋值a[:]相当于复制了a对应的那段空 
b is a False
b == a True
```

#### 8、运算符优先 

```python
运算     描述






```





(expressions...),

[expressions...], {key: value...}, {expressions...}

圆括号的表达 

x[index], x[index:index], x(arguments...), x.attribute

读取，切片，调用，属性引 

await x

await 表达 

**

乘方(指数)

+x, -x, ~x

正，负，按位  NOT

*, @, /, //, %

乘，矩阵乘，除，整除，取 

+, -

加和 

<<, >>

移位

&

按位  AND

^

按位异或 XOR

|

按位  OR

in,not in, is,is not, <, <=, >, >=, !=, ==

比较运算，包括成员检测和标识号检 

not x

逻辑  NOT

and

逻辑  AND

or

逻辑  OR

if -- else

条件表达 

lambda

lambda 表达 

:=

赋值表达式

## 第三章  数字函数

几个常量 
e：math.e
pi:math.pi

### 一、数学函数 

#### 1、abs(x) 

返回数字的绝对值，复数返回 

#### 2、ceil(x) 

返回数字的上取整 (注意要调用import math)

用法 

```python
import math
math.ceil()
```

#### 3、比较函数

((x>y)-(x<y)

*   x < y, 返回-1  若x > y, 返回1  若x == y, 返回0

#### 4、exp(x) 

返回e的x次幂(注意要调用import math)

#### 5、floor(x) 

返回数字的下取整 (注意要调用import math)

#### 6、log(x) 

返回x的自然对 (注意要调用import math)

#### 7、log10(x) 

返回10为底的对 (注意要调用import math)

#### 8、max(x,y,z) 

返回最大的数 (min(x,y,z)同理)

* 多序列入参，按索引顺序，逐一对比各序列当前索引位的值，直到遇见最大值立即停止对比，并返回最大值所在序 

* True当做1，False当做0

#### 9、modf() 

返回x的整数部分和小数部分，两部分的数值符号与x相同，整数部分以浮点型表 (注意要调用import math)

#### 10、pow(x,y) 

返回x的y次方(注意要调用import math)

* 结果是float 
* 注意：直接调用的pow(x,y),结果为int 

#### 11、round(x,y) 

保留x的小数点到y位，采取四舍五入的方法

ps  精度不好掌控，尽量别 

#### 12、sqrt(x) 

返回x的float型的平方 

### 二、随机数函数(注意调用import random)

#### 1、choice(seq) 

返回一个列表，元组或字符串的随机项

```python
import random 
radom.choice(seq)  #seq-> range[100], [1,2,3,4,5], 'numpy'...
```

```python
random password
import string #string module里包含了阿拉伯数 ,ascii ,特殊符号
import random #需要利用到choice

a = int(input('请输入要求的密码长度'))
b = string.digits + string.ascii_letters + string.punctuation #构建密码 
password = "" #命名一个字符串

for i in range(0,a):  #for loop 指定重复次数
    password = password + random.choice(b)   #从密码池中随机挑选内容构建密 
print(password)   #输出密码
```

#### 2、randrange(x,y,z) 

从左闭右开的区间[x,y)内以步长为z选取随机  randrange(x) 从左闭右开的区间[0,y)内选取随机 

* randrange(1,100, 3)  #  0-100中随机选取一个能 3整除后余1的数
* 注意y的值不包含在内

#### 3、seed()

```python
import random

random.seed()
print ("使用默认种子生成随机数：", random.random())
print ("使用默认种子生成随机数：", random.random())

random.seed(10)
print ("使用整数 10 种子生成随机数：", random.random())
random.seed(10)
print ("使用整数 10 种子生成随机数：", random.random())
>>>
使用默认种子生成随机数： 0.7908102856355441
使用默认种子生成随机数： 0.81038961519195
使用整数 10 种子生成随机数： 0.5714025946899135
使用整数 10 种子生成随机数： 0.5714025946899135

```

#### 4、shuffle(seq) 

随机排序列表

```python
import random
list = [1,2,3,4]
random.shuffle(list)
print(list)
>>>[2,3,1,4]
```

#### 5、uniform(x,y) 

返回浮点数，且范围是如果 x<y   x <= N <= y，如  y<x 则y <= N <= x

### 三、三角函数 

```python
cos(x) acos(x)
sin(x) asin(x)
tan(x) atan(x) atan2(x,y) 返回给定的x及y坐标的反正切 
hypot(x,y) 返回欧几里得范数 sqrt(x**2,y**2)
degrees(x) 弧度转化为角 
radians(x) 角度转化为弧 
```

## 第四章  字符串函数

### 一、字符串格式化函数str.fomat()

### 二、str.split(str =' ', num)

* 用于拆掉字符串中间空白字符，返回一个列表

* str-分隔符，默认为所有的空字符(' ', \n, \t)

* num-分割num+1个字符串。默认为-1，即分割所有

### 三、join

去掉字符串中的标点

```python
import string
 
def removePunctuation(text):
    '''去掉字符串中标点符号
    '''
    #方法一：使用列表添加每个字符，最后将列表拼接成字符串，目测要五行代码以上
    temp = []
    for c in text:
        if c not in string.punctuation:
            temp.append(c)
    newText = ''.join(temp)
    print(newText)
 
    #方法二：给join传递入参时计算符合条件的字符
    b = ''.join(c for c in text if c not in string.punctuation)
    print(b)
    return newText
```





## 第五章 列表函数

### 一、 list.append(element)

插入元素到末尾

```python

```



### 二、 list.insert(i，str)

将元素str插入到索引号为i位置

### 三、  list.pop()

删除末尾元素

### 四、  list.pop(i)

删除在i位置的元素

### 五、 list[i] = str

将序号为i的位置元素替换成str

## 第六章  元组

## 第七章  字典

### 一、len(dict)

访问字典键的数目

### 二、修改字 -直接对值进行赋 

### 三、删除字典元 

```python
del dict[key] #删除 
dict.clear()  #清空字典内所有元 
del dict      #删除字典
```

### 四、dict.copy() 

返回一个字符串的浅复制

dict2 = dict1        #浅拷贝：引用对象 (相当于给dict1起了一个别名dict2)
dict3 = dict1.copy() #深拷贝：拷贝了一份dict1的内容，dict3不会随dict1的改变而改 

### 五、dict.pop(key) 

删除key和value

## 第八章  集合

### 一、set.add(value) 

添加元素到set 

### 二、set.remove(value) 

删除元素

## 第九章  条件判断

* if 归根到底判断的是后面表达式的BOOL值

1、注意else后面要加`:`

​     地位相同的执行语句要缩进相同格数

```python
if xxx:
  print
elif xxx:
  print
else 
  print
```

2、简写：
x是非0数值，非空字符串、非空list，就判定为True，否则为False

```python
if x:
  print
```

3、match...case

```python
match subject
    case <pattern_1>
       <action_1>
    case <pattern_2>
       <action_2>
    case _:
        <action_wildcard> #case _可以匹配一切，当其他case无法匹配时，匹配这条可以保证匹配成功
```

## 第十章  循环

1、for...in 循环(依次将list或tuple中元素迭代出 )

```python
#实现求和
sum = 0
for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
for x in range(1,11)：  #使用range函数
    sum = sum + x   
print(sum)
```

* range(start, stop, step) 可用于创建一个整数列 
  从start开始，stop结束(不包括stop)，步长为step
  PS：如果step是正数，则从小到大**递增**
    如果step是负数，则从大到小**递减**
* range(stop)  0-stop-1

* range(start,stop)

```python
#输出数对
x = [(1,2), (3,4), (5,6)]
for (a,b) in x:
   print(a,b)
```


```python
for num in [1,2,3,4,5]
   print('Hello')   #实现多次输出
```

2、while循环

while (判断语句)
直到条件不满足时退出循 

3、break

提前退出循环   

```python
n = 1
while n <= 100:
    if n > 10: # 当n = 11时，条件满足，执行break语句
        break # break语句会结束当前循 
    print(n)
    n = n + 1
print('END')
```

4、continue

跳过当次循环，直接开始下次循 

```python
n = 0
while n < 10:
    n = n + 1
    if n % 2 == 0: # 如果n是偶数，执行continue语句
        continue # continue语句会直接继续下一轮循环，后续的print()语句不会执行
    print(n)
```

## 第十一章 函数

### 一、定义函数 

def 函数 (参数列表)
    函数 
return选择性地返回一个值给调用方，若不带表达式的return相当于返回None

```python
def func(a,b): #括号内为自变 
  <action_a>
     return a
   <action_b>
     return b
```

### 二、函数调用

1、操作型 
直接调用func()

2、传值型 
func()看作变量

### 三、参数传值

1、不可变类型 
int、string、tuple
fun(a),传递的是a的值，不改变a对象本身，若在fun(a)内部修改a的值不影响a在函数外部的 

2、可变类型：
list、dict
fun(a)，是将a真正传递过去，若在fun(a)内部修改a的值，则在fun(a)的外部也会修改a的 

## 四、参数 

### 1、关键字参数 

参数使用时不需要按照声明时的指定顺 

```python
def printinfo( name, age ):
   "打印任何传入的字符串"
   print ("名字: ", name)
   print ("年龄: ", age)
   return
 
#调用printinfo函数
printinfo( age=50, name="runoob" )
```

### 2、默认参数：

调用函数时，如果没有传递参数，则会使用默认参数

```python
#可写函数说明
def printinfo( name, age = 35 ):
   "打印任何传入的字符串"
   print ("名字: ", name)
   print ("年龄: ", age)
   return
 
#调用printinfo函数
printinfo( age=50, name="runoob" )
print ("------------------------")
printinfo( name="runoob" )
```

### 3、不定长参数 

* 加了`*`的参数会以元组的形式导入，存放所有未命名的变量参 

```python
def printinfo( arg1, *vartuple ):
   "打印任何传入的参 "
   print ("输出: ")
   print (arg1)
   print (vartuple)
 
# 调用printinfo 函数
printinfo( 70, 60, 50 )
```

* 如果函数调用时没有指定参数，则其为空元组，不输出 

```python
def printinfo( arg1, *vartuple ):
   "打印任何传入的参 "
   print ("输出: ")
   print (arg1)
   for var in vartuple:
      print (var)
   return
 
# 调用printinfo 函数
printinfo( 10 )   #后面没有元组对应的参数，不输 
printinfo( 70, 60, 50 )
```

* 带有`**`的参数会以字典形式导 

```python
def printinfo( arg1, **vardict ):
   "打印任何传入的参 "
   print ("输出: ")
   print (arg1)
   print (vardict)
 
# 调用printinfo 函数
printinfo(1, a=2,b=3)

>>>输出 
1
{'a': 2, 'b': 3}
```

* `*`可以单独出现，其后面的参数必须用关键字传值 

```python
def f(a,b,*,c)
```

## 第十二章 模块

### 一、定义



### 二、模块引入

#### 1、import

* import module1...

* 使用模块中的函数时，需要module.function形式

#### 2、from...import

```python
from module import function 
          |||
import module
module.function
```

* 从模块中导入一个函数,调用该函数时不用加上模块的前缀，直接调用函数名即可

#### 3、from...import*

* 把一个模块中的所有内容全部导入到当前的命名空间

### 三、导入包

* 第一种情况：包和代码在一个文件夹下

1、创建文件夹，用于存放相关的模块，文件夹的名字即包的名字

2、在文件夹中创建一个```_init_.py``` 模块文件，内容可以为空

3、将相关模块导入到文件夹中

```python
#导入模块
import 包名.模块名

#导入函数
import 包名.模块名.函数名
#or
from 包名.模块名 import
```



* 第二种情况：不在一个文件夹下

将导入的模块放入Pythonxx\lib\site-packages路径下

```python
import sys
sys.path # python导入package时自动搜索的路径
sys.path.append(r"C:\Users\apple\Desktop\Research\SEIM_v2.0_20200927\run(core)\module")
sys.path # 添加了以上路径
```











## 第十三章、正则表达式

### 一、简介

正则表达式包括普通字符和特殊字符(元字符)

正则表达式使用单个字符串来描述、匹配一系列匹配某个句法规则的字符串

### 二、使用字符描述字符

#### 1、基本格式：

* `\d`可以匹配一个数字
* `\w`可以匹配一个数字或字母
* `.`可以匹配任意字符
* `*`可以匹配任意个字符(包括0个)
* `+` 至少一个字符
* `?` 匹配0或1个字符
* `{n}`匹配n个字符
* `{n,m}`匹配n-m个字符
* `A|B`匹配A或B
* `\-`匹配-
* `\_`匹配_

#### 2、精确匹配：

* `[0-9a-zA-Z\_]`匹配一个数字、字母或下划线
* `[0-9a-zA-Z\_]+`匹配至少有一个字母、数字或下划线组成的字符串
* `[a-zA-Z\_][0-9a-zA-Z\_]*`匹配由字母或下划线开头，后接任意个由一个数字或字母或下划线组成的字符串——python合法变量
* `[a-zA-Z\_][0-9a-zA-Z\_]{0, 19}`在上述基础上限制了变量的长度是1-20个字符


* `^`表示行的开头 
* `^\d`表示必须以数字开头 
* `$`表示行的结束 
* `\d$`表示必须以数字结束    #两者均需打在()外面

#### 3、匹配字符串：

* 注意转义：字符串前面加上r，不用考虑\转义

* 判断表达式是否匹配：

```python
import re
test = '用户输入的字符串'
if re.match(r'正则表达式', test):
   print('ok')   #匹配成功，返回match对象
else:
   print('failed')   #匹配不成功返回None
```

#### 4、切分字符串：

按照能够匹配的子串(此串相当于要剔除的)将字符串分割后返回列表(可以识别连续空格)

```python
import re
re.split(r'\s+', 'a b  c') #返回['a', 'b', 'c']
re.split(r'[\s\,\;]+','a,  b,;c  ')
```

#### 5、分组：

(提取子串)
用()表示的就是要提取的子串

```python
m = re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')
m.group(0) = '010-12345'    #group(0)提取出与整个正则表达式相匹配的字符串
m.group(1) = '010'          #group(n)提取出第n个子串
m.group(2) = '12345'   
m.groups() = ('010', '12345') #groups()提取各组子串，返回元组
```

#### 6、贪婪匹配：

匹配尽可能多的字符；故要实现非贪婪匹配，需要加上`？`

```python
re.match(r'^(\d+)(0*)$', '102300').groups()
>>>('102300', '')
re.match(r'^(\d+?)(0*)$').groups()
>>>('10230', '00')
```

#### 7、编译：

```python
import re
#编译:
re_fun = re.compile(r'^(\d{3})-(\d{3,8})$')
#使用:
re_fun.match('010-12345').groups()
```



##  第十四章 类和实例

一、类
1、
```python
#定义实例
class Dog:
    def __init__(self, name, age): #注意这个self永远是第一个参数，在类里面看作一泛化的实例
        # 初始化属性 name 和 age
        self.name = name
        self.age = age

    def bark(self):#此处定义了一个类的方法
        print(f"{self.name} says: Woof!")

    def get_age(self): #此处定义了一个类的属性-年龄
        return self.age

# 创建 Dog 类的实例
my_dog = Dog("Buddy", 3) #此时Buddy和3就传入了name和age中

# 访问属性
print("My dog's name is:", my_dog.name)
print("My dog's age is:", my_dog.get_age())

# 调用方法
my_dog.bark()


```







### * 一些命令行小操作：





# 数据结构

## 第一章 数组

### 一、



## 第二章 链表

### 一、定义

![链表](https://qcdn.itcharge.cn/images/202405092229936.png)

链表通过将一组任意的存储单元串联在一起。其中，每个数据元素占用若干存储单元的组合称为一个「链节点」



### 二、链表操作

1、结构定义：

* next：后继指针，用于连接链节点

```python
# 链节点类(成员变量val表示数据元素的值，指针变量next表示后继指针)
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 链表类(链节点变量head表示链表的头节点，设置时设为空链接)
class LinkedList:
    def __init__(self):
        self.head = None
```



2、建立线性链表：

* cur：当前指针节点

```python
# 根据 data 初始化一个新链表
def create(self, data):
    if not data:
        return
    self.head = ListNode(data[0]) #从线性表中取出第一个数据元素
    cur = self.head #将这个元素设为链表头节点
    for i in range(1, len(data)):
        node = ListNode(data[i]) #获取新的数据元素
        cur.next = node          #新节点插入到链表尾部
        cur = cur.next           #当前节点变为新的节点
```



3、求线性链表的长度：

```python
# 获取线性链表长度
def length(self):
    count = 0
    cur = self.head #cur指向链表的第一个节点
    while cur:
        count += 1  #cur每指向一个链节点，计数器加1
        cur = cur.next #顺着链节点的next指针遍历链表
    return count
```



4、查找元素：

```python
# 查找元素：在链表中查找值为 val 的元素
def find(self, val):
    cur = self.head	#cur指向链表的第一个节点
    while cur:
        if val == cur.val: #遇到当前节点的值等于要查找的值，返回当前指针变量cur
            return cur
        cur = cur.next	#顺着链节点的next指针遍历链表

    return None

```



5、插入元素：

（1）头部插入：O(1)

```python
# 链表头部插入元素
def insertFront(self, val):
    node = ListNode(val) #创建一个链节点node
    node.next = self.head #将node的next指针指向链表的头节点head
    self.head = node #将链表的头节点head指向node
```

![链表头部插入元素](https://qcdn.itcharge.cn/images/202405092231514.png)



（2）尾部插入：O(n)

```python
# 链表尾部插入元素
def insertRear(self, val):
    node = ListNode(val) 
    cur = self.head		#cur指向头节点head
    while cur.next:		#移动next指针从而遍历链表，直到cur.next为None
        cur = cur.next	
    cur.next = node		#令cur.next指向新的链节点node
```



（3）中间插入元素（第i个链节点之前插入值为val的链节点）：O(n)

```python
# 链表中间插入元素
def insertInside(self, index, val):
    count = 0
    cur = self.head
    while cur and count < index - 1: #沿着next指针遍历链表，遍历到第index-1个链节点停止遍历
        count += 1
        cur = cur.next
        
    if not cur:
        return 'Error'
     
    node = ListNode(val)
    node.next = cur.next
    cur.next = node
```

![链表中间插入元素](https://qcdn.itcharge.cn/images/202405092232900.png)



6、改变元素：O(n)

```python
# 改变元素：将链表中第 i 个元素值改为 val
def change(self, index, val):
    count = 0
    cur = self.head
    while cur and count < index:
        count += 1
        cur = cur.next
        
    if not cur:
        return 'Error'
    
    cur.val = val
```



7、删除元素：

（1）删除头部元素：O(1)

```python
# 链表头部删除元素
def removeFront(self):
    if self.head:
        self.head = self.head.next #将self.head沿着指针向右移动一格
```

（2）
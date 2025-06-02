/* 题目（质数）：找2-100之间的素数 */

# include <stdio.h>

int main(void)
{
	int i, k, tag;	
	
	for (i=2; i<=100; i++)	//i是1-100之间的数
	{
		tag = 0;	//tag表示标志

		for (k=2; k<i; k++)	//k是1到i之间的任意一个数
		{
			if (i%k == 0)	//如果i被k整除，则tag=1，i不是质数，i不被输出
				tag = 1;
		}
		if (tag == 0)		
			printf("%d,", i);	//d后面加“,”则输出结果会被逗号隔开
	}
	
	return 0;
}

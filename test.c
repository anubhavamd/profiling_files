#include <stdio.h>
 
void func2()
{
   int count = 0;
   for(count=0; count < 0XFFFFF; count++);
 
   return;
}
 
void func1(void)
{
   int count = 0;
   for(count=0; count < 0XFF; count++)
       func2();
 
   return;
}
 
int main(void)
{
    printf("\n Hello World! \n");
    func1();
    func2();
    return 0;
}

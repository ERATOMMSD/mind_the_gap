# 1 "01x.c"
# 1 "<built-in>" 1
# 1 "<built-in>" 3
# 361 "<built-in>" 3
# 1 "<command line>" 1
# 1 "<built-in>" 2
# 1 "01x.c" 2
void errorFn() {assert(0);}
int unknown1();
int unknown2();
int unknown3();
int unknown4();





void main()
{
  int z;
 int x=1; int y=1;
 while(unknown1()) {
   int t1 = x;
   int t2 = y;
   x = t1+ t2;
   y = t1 + t2;
 }
 if(!(x>=y))
  errorFn();
}

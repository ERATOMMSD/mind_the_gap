void errorFn() {assert(0);}
int unknown1();
int unknown2();
int unknown3();
int unknown4();


void main()
{
    int x=1; 
    x = x + 1;
    if(x % 2 == 1)
    errorFn();
}


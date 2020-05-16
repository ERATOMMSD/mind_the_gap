void errorFn() {assert(0);}
int unknown1();
int unknown2();
int unknown3();
int unknown4();


void main()
{
    int x=0; 
    while(unknown1()){
        x = x + 3;
    }
    if(x % 3 != 0)
    errorFn();
}


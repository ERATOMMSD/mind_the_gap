void errorFn() {assert(0);}
int unknown1();
int unknown2();
int unknown3();
int unknown4();


void main()
{
    int x=0; 
    int i=0;
    while(unknown1()){
        i = i + 1;
        if (i%2 == 0) x = x + 1;
    }
    if(i %2 == 0){
        if(2*x != i + 1){
            errorFn();
        }
    }
}




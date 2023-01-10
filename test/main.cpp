#include <iostream>
#include "test/test_data/read_data.h"
int main(int argc, char **argv) {

  int a[5] = {1,1,2,2,3};

  write_date("../../test/test_data/input.dat",5,a);

  int result[5];
  read_date("../../test/test_data/input.dat",result);
  for(int i=0; i<5;i++){
    std::cout << result[i] << std::endl;
  }
  return  0;
 }
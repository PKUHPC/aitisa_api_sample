#include <fstream>
#include <iostream>
template <class datetype>
void read_date(char* path, datetype* a) {
  int num;
  std::ifstream infile;
  infile.open(path, std::ios::in);
  if (!infile.is_open()) {
    std::cout << "error" << std::endl;
    exit(1);
  }
  infile >> num;
  for (int i = 0; i < num; i++) {
    infile >> a[i];
  }
  infile.close();
}

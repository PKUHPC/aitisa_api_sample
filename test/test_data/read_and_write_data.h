#include <fstream>
#include <iostream>
template <class datetype>
void read_date(std::string path, datetype* a) {
  int num;
  std::ifstream infile;
  infile.open(path, std::ios::in);
  if (!infile.is_open()) {
    std::cout << "read error" << std::endl;
    exit(1);
  }
  infile >> num;
  for (int i = 0; i < num; i++) {
    infile >> a[i];
  }
  infile.close();
}

template <class datetype>
void write_date(std::string path, int num, datetype* a) {

  std::ofstream outfile;
  outfile.open(path, std::ios::out);
  if (!outfile.is_open()) {
    std::cout << "write error" << std::endl;
    exit(1);
  }
  outfile << num << std::endl;
  for (int i = 0; i < num; i++) {
    outfile << a[i] << " ";
  }
  outfile.close();
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "starfish.hpp"
namespace py=pybind11;
PYBIND11_MODULE(starfish_native,m){
 m.def("crossing_numbers",[](py::array_t<uint8_t,py::array::c_style|py::array::forcecast>a){
  auto b=a.request(); if(b.ndim!=2) throw std::runtime_error("expected 2D array");
  int h=(int)b.shape[0],w=(int)b.shape[1]; auto v=starfish::crossing_numbers((uint8_t*)b.ptr,w,h);
  py::array_t<int> out({h,w}); auto o=out.mutable_unchecked<2>();
  for(int y=0;y<h;y++)for(int x=0;x<w;x++)o(y,x)=v[(size_t)y*w+x]; return out;
 });
}
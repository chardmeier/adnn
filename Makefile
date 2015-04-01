#CXX = icpc
CXX = g++

#HOME = /cluster/home/chm
HOME = /tmp/chm2

BOOST = $(HOME)/boost_1_54_0
ADEPT = $(HOME)/adept-1.0
EIGEN = $(HOME)/eigen-3.2.4 

nnopt:	3layer.h nnopt.cpp
	$(CXX) -std=c++14 -o 3layer -g -O3 -Wall -Wno-unused-local-typedefs -I$(BOOST) -I$(EIGEN) -I$(ADEPT)/include -L$(ADEPT)/lib nnopt.cpp -ladept -lm

clean:
	rm nnopt

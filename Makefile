#OPT = -O3
OPT =

#CXX = icpc
CXX = g++

#HOME = /cluster/home/chm
HOME = /tmp/chm2

BOOST = $(HOME)/boost_1_57_0
ADEPT = $(HOME)/adept-1.0
EIGEN = $(HOME)/eigen-3.2.4 

NNET_HEADERS = nnet.h net_wrapper.h nnopt.h mlp.h logbilinear_lm.h

lmtest:	lmtest.cpp $(NNET_HEADERS)
	$(CXX) -std=c++14 -ftemplate-backtrace-limit=0 -o 3layer -g $(OPT) -Wall -Wno-unused-local-typedefs -I$(BOOST) -I$(EIGEN) -I$(ADEPT)/include -L$(ADEPT)/lib lmtest.cpp -ladept -lm

nnopt:	3layer.cpp $(NNET_HEADERS)
	$(CXX) -std=c++14 -o 3layer -g $(OPT) -Wall -Wno-unused-local-typedefs -I$(BOOST) -I$(EIGEN) -I$(ADEPT)/include -L$(ADEPT)/lib 3layer.cpp -ladept -lm

clean:
	rm nnopt

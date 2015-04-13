#OPT = -O3
OPT =

#CXX = icpc
#CXX_FLAGS = -std=c++14 -DBOOST_RESULT_OF_USE_DECLTYPE -Wall -Wno-comment

CXX = g++
CXX_FLAGS = -std=c++14 -ftemplate-backtrace-limit=0 -Wall -Wno-unused-local-typedefs -march=native

#ADEPT_FLAGS = -DADEPT_INITIAL_STACK_LENGTH=2000000000 
ADEPT_FLAGS = -DADEPT_INITIAL_STACK_LENGTH=100000

#MKL = 
MKL = -DEIGEN_MKL_USE_ALL -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group \
	-liomp5 -ldl -lpthread -lm \
	-DMKL_ILP64 -m64 -I${MKLROOT}/include

HOME = /cluster/home/chm
#HOME = /tmp/chm2

BOOST = $(HOME)/boost_1_57_0
ADEPT = $(HOME)/adept-1.0
EIGEN = $(HOME)/eigen-3.2.4 

NNET_HEADERS = nnet.h net_wrapper.h nnopt.h mlp.h logbilinear_lm.h

lmtest:	lmtest.cpp $(NNET_HEADERS)
	$(CXX) $(CXX_FLAGS) $(MKL) $(ADEPT_FLAGS) -o lmtest -g $(OPT) -I$(BOOST) -I$(EIGEN) -I$(ADEPT)/include -L$(ADEPT)/lib lmtest.cpp -ladept -lm

nnopt:	3layer.cpp $(NNET_HEADERS)
	$(CXX) -std=c++0x -o 3layer -g $(OPT) -Wall -Wno-unused-local-typedefs -I$(BOOST) -I$(EIGEN) -I$(ADEPT)/include -L$(ADEPT)/lib 3layer.cpp -ladept -lm

clean:
	rm -f lmtest nnopt

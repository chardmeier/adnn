#OPT = -O3 -march=native -ffast-math # -fopenmp
OPT =

#CXX = icpc
#CXX_FLAGS = -std=c++14 -DBOOST_RESULT_OF_USE_DECLTYPE -Wall -Wno-comment

CXX = g++
CXX_FLAGS = -std=c++14 -ftemplate-backtrace-limit=0 -Wall -Wno-unused-local-typedefs -Wno-deprecated-declarations -Wno-return-type

#MKL = 
MKL = -DEIGEN_MKL_USE_ALL -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group \
	-liomp5 -ldl -lpthread -lm \
	-DMKL_ILP64 -m64 -I${MKLROOT}/include

HOME = /cluster/home/chm
#HOME = /tmp/chm2

BOOST = $(HOME)/boost_1_57_0
EIGEN = $(HOME)/eigen-3.2.4 

NNET_HEADERS = nnet.h nnopt.h mlp.h logbilinear_lm.h nn6.h nn6-dev.h

compile: nn6

lblm: lblm.cc netops.h $(NNET_HEADERS)
	$(CXX) $(CXX_FLAGS) $(OPT) $(MKL) -o lblm -g -I$(BOOST) -I$(EIGEN) lblm.cc -lm

nn6-dev: nn6-dev.cc netops.h $(NNET_HEADERS)
	$(CXX) $(CXX_FLAGS) $(OPT) $(MKL) -o nn6-dev -g -I$(BOOST) -I$(EIGEN) nn6-dev.cc -lm

nn6: nn6.cc netops.h $(NNET_HEADERS)
	$(CXX) $(CXX_FLAGS) $(OPT) $(MKL) -o nn6 -g -I$(BOOST) -I$(EIGEN) nn6.cc -lm

nn6-cmp: nn6-cmp.cc netops.h $(NNET_HEADERS)
	$(CXX) $(CXX_FLAGS) $(OPT) $(MKL) -o nn6-cmp -g -I$(BOOST) -I$(EIGEN) nn6-cmp.cc -lm

nn6_gradient_check: nn6_gradient_check.cc netops.h $(NNET_HEADERS)
	$(CXX) $(CXX_FLAGS) $(OPT) $(MKL) -o nn6_gradient_check -g -I$(BOOST) -I$(EIGEN) nn6_gradient_check.cc -lm

gradient_check: gradient_check.cc netops.h $(NNET_HEADERS)
	$(CXX) $(CXX_FLAGS) $(OPT) $(MKL) -o gradient_check -g -I$(BOOST) -I$(EIGEN) gradient_check.cc -lm

netops:	netops.cc netops.h
	$(CXX) $(CXX_FLAGS) $(OPT) $(MKL) -o netops -g $(OPT) -I$(BOOST) -I$(EIGEN) netops.cc -lm

ptrtst: ptrtst.cc
	$(CXX) $(CXX_FLAGS) $(OPT) -g -I$(EIGEN) -o ptrtst ptrtst.cc

lmtest:	lmtest.cpp $(NNET_HEADERS)
	$(CXX) $(CXX_FLAGS) $(OPT) $(MKL) -o lmtest -g $(OPT) -I$(BOOST) -I$(EIGEN) lmtest.cpp -lm

3layer:	3layer.cc $(NNET_HEADERS)
	$(CXX) $(CXX_FLAGS) -o 3layer -g $(OPT) -Wall -Wno-unused-local-typedefs -I$(BOOST) -I$(EIGEN) 3layer.cc -lm

clean:
	rm -f lmtest 3layer netops ptrtst gradient_check nn6_gradient_check nn6 lblm

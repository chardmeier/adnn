#undef _GNU_SOURCE
#define _GNU_SOURCE
#include <fenv.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/vector_c.hpp>

#include "nnet.h"
#include "netops.h"

namespace fusion = boost::fusion;
namespace mpl = boost::mpl;

typedef double Float;
typedef unsigned long voc_id;
typedef Eigen::Matrix<Float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> matrix;

enum { CE, CELA, ELLE, ELLES, IL, ILS, ON, CA, OTHER, NCLASSES };

template<class Idx,int WB,class Net,class Input,class Weights,class Targets>
void check(const Net &net, const Input &input, const Weights &ww, const Weights &grad, const Targets &targets, int i, int j) {
	const Float EPS = 1e-4;

	typedef typename mpl::push_back<typename mpl::pop_front<Idx>::type,mpl::int_<WB>>::type WIdx;

	Weights disturb(ww);
	Weights xgrad(ww);

	Float ow = netops::at_spec<WIdx>(ww.sequence())(i,j);

	netops::at_spec<WIdx>(disturb.sequence())(i,j) = ow + EPS;
	matrix out1 = net.fprop(fusion::make_vector(input, disturb.sequence()));
	Float j1 = -targets.cwiseProduct(out1.array().log().matrix()).sum();

	netops::at_spec<WIdx>(disturb.sequence())(i,j) = ow - EPS;
	matrix out2 = net.fprop(fusion::make_vector(input, disturb.sequence()));
	Float j2 = -targets.cwiseProduct(out2.array().log().matrix()).sum();

	Float g = (j1 - j2) / (2 * EPS);

	std::cout << "< ";
	boost::mpl::for_each<WIdx>([] (auto x) { std::cout << decltype(x)::value << ' '; });
	std::cout << ">(" << i << ',' << j << "): " << g << " == " <<
		netops::at_spec<WIdx>(grad.sequence())(i,j) << std::endl;
}

struct vocmap {
	typedef std::unordered_map<std::string,voc_id> map_type;

	voc_id maxid;
	map_type map;

	vocmap() : maxid(1) {
		map.insert(std::make_pair("<unk>", 0));
	}
};

voc_id voc_lookup(const std::string &word, vocmap &voc, bool extend) {
	voc_id id;
	vocmap::map_type::const_iterator it = voc.map.find(word);
	if(it != voc.map.end()) 
		id = it->second;
	else {
		if(extend) {
			id = voc.maxid++;
			voc.map.insert(std::make_pair(word, id));
		} else
			id = 0;
	}

	return id;
}

auto load_nn6(const std::string &file, vocmap &srcvocmap, vocmap &antvocmap) {
	bool make_vocabulary = srcvocmap.map.size() <= 1;

	std::ifstream is(file);
	if(!is.good())
		throw 0;

	std::size_t nexmpl = 0;
	std::size_t nant = 0;
	std::size_t nlink = 0;

	std::vector<std::string> nn6_lines;
	for(std::string line; getline(is, line);) {
		if(line.compare(0, 8, "ANAPHORA") == 0)
			nexmpl++;
		else if(line.compare(0, 10, "ANTECEDENT") == 0)
			nant++;
		else if(line.compare(0, 2, "-1") == 0) {
			std::istringstream ss(line);
			std::size_t idx;
			char colon;
			Float val;
			ss >> idx; // get rid of -1 in the beginning
			while(ss >> idx >> colon >> val)
				if(idx > nlink)
					nlink = idx;
		}
		nn6_lines.push_back(line);
	}

	typedef Eigen::Matrix<Float,Eigen::Dynamic,1> vector;
	typedef Eigen::Matrix<voc_id,Eigen::Dynamic,1> vocid_vector;
	typedef Eigen::Matrix<int,Eigen::Dynamic,1> int_vector;

	typedef std::unordered_map<std::string,int> classmap_type;
	classmap_type classmap;

	classmap.insert(std::make_pair("ce", CE));
	classmap.insert(std::make_pair("c'", CE));
	classmap.insert(std::make_pair("cela", CELA));
	classmap.insert(std::make_pair("elle", ELLE));
	classmap.insert(std::make_pair("elles", ELLES));
	classmap.insert(std::make_pair("il", IL));
	classmap.insert(std::make_pair("ils", ILS));
	classmap.insert(std::make_pair("on", ON));
	classmap.insert(std::make_pair("ça", CA));
	classmap.insert(std::make_pair("ca", CA));
	classmap.insert(std::make_pair("ç'", CA));

	matrix targets(nexmpl, static_cast<int>(NCLASSES));
	vector nada(nexmpl);
	matrix T(nant, nlink);
	int_vector antmap(nexmpl);
	std::vector<Eigen::Triplet<Float>> ant_triplets;
	std::vector<std::vector<Eigen::Triplet<Float>>> srcctx_triplets(7);

	nada.setZero();
	T.setZero();
	antmap.setZero();
	targets.setZero();

	for(std::size_t i = 0, ex = std::numeric_limits<std::size_t>::max(), ant = std::numeric_limits<std::size_t>::max();
			i < nn6_lines.size(); i++) {
		std::istringstream ss(nn6_lines[i]);
		std::string tag;
		getline(ss, tag, ' ');
		if(tag == "ANAPHORA") {
			if(ex < nexmpl)
				antmap(ex) = ant + 1;

			ex++;
			std::string word;
			for(int j = 0; j < 7; j++) {
				getline(ss, word, ' ');
				voc_id v = voc_lookup(word, srcvocmap, make_vocabulary);
				srcctx_triplets[j].push_back(Eigen::Triplet<Float>(ex, v, 1));
			}
			ant = std::numeric_limits<std::size_t>::max();
		} else if(tag == "TARGET") {
			std::string word;
			bool found = false;
			while(getline(ss, word, ' ')) {
				boost::to_lower(word);
				classmap_type::const_iterator it = classmap.find(word);
				if(it != classmap.end()) {
					targets(ex, it->second)++;
					found = true;
				}
			}
			if(!found)
				targets(ex, OTHER) = 1;
		} else if(tag == "NADA")
			ss >> nada(ex);
		else if(tag == "ANTECEDENT") {
			ant++;
			int nwords = std::count(nn6_lines[i].begin(), nn6_lines[i].end(), ' ');
			std::string word;
			while(getline(ss, word, ' ')) {
				voc_id v = voc_lookup(word, antvocmap, make_vocabulary);
				ant_triplets.push_back(Eigen::Triplet<Float>(ant, v, 1.0 / nwords));
			}
		} else if(tag == "-1") {
			int fidx;
			char colon;
			Float fval;
			while(ss >> fidx >> colon >> fval)
				T(ant, fidx - 1) = fval;
		}
	}

	targets.array().colwise() /= targets.array().rowwise().sum();

	Eigen::SparseMatrix<Float,Eigen::RowMajor> A(nant, antvocmap.map.size());
	A.setFromTriplets(ant_triplets.begin(), ant_triplets.end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> L3(nexmpl, srcvocmap.map.size());
	L3.setFromTriplets(srcctx_triplets[0].begin(), srcctx_triplets[0].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> L2(nexmpl, srcvocmap.map.size());
	L2.setFromTriplets(srcctx_triplets[1].begin(), srcctx_triplets[1].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> L1(nexmpl, srcvocmap.map.size());
	L1.setFromTriplets(srcctx_triplets[2].begin(), srcctx_triplets[2].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> P(nexmpl, srcvocmap.map.size());
	P.setFromTriplets(srcctx_triplets[3].begin(), srcctx_triplets[3].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> R1(nexmpl, srcvocmap.map.size());
	R1.setFromTriplets(srcctx_triplets[4].begin(), srcctx_triplets[4].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> R2(nexmpl, srcvocmap.map.size());
	R2.setFromTriplets(srcctx_triplets[5].begin(), srcctx_triplets[5].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> R3(nexmpl, srcvocmap.map.size());
	R3.setFromTriplets(srcctx_triplets[6].begin(), srcctx_triplets[6].end());

	return std::make_pair(
		fusion::make_vector(fusion::make_vector(A, T, antmap),
			fusion::make_vector(L3, L2, L1, P, R1, R2, R3)),
		targets);
}

auto lweights(std::size_t rows, std::size_t cols) {
	return fusion::vector2<nnet::mat_size<Float>,nnet::vec_size<Float>>(
		nnet::mat_size<Float>(rows, cols), nnet::vec_size<Float>(cols));
}

int main() {
	feenableexcept(FE_INVALID | FE_DIVBYZERO);

	vocmap srcvocmap;
	vocmap antvocmap;
	//auto data = load_nn6("/work/users/chm/ncv9.nn6", srcvocmap, antvocmap);
	auto data = load_nn6("small.nn6", srcvocmap, antvocmap);
	const auto &input = data.first;
	const auto &targets = data.second;
	std::cerr << "Data loaded." << std::endl;

	typedef mpl::vector3_c<int,0,0,0> I_A;
	typedef mpl::vector3_c<int,0,0,1> I_T;
	typedef mpl::vector3_c<int,0,0,2> I_antmap;

	typedef mpl::vector3_c<int,0,1,0> I_L3;
	typedef mpl::vector3_c<int,0,1,1> I_L2;
	typedef mpl::vector3_c<int,0,1,2> I_L1;
	typedef mpl::vector3_c<int,0,1,3> I_P;
	typedef mpl::vector3_c<int,0,1,4> I_R1;
	typedef mpl::vector3_c<int,0,1,5> I_R2;
	typedef mpl::vector3_c<int,0,1,6> I_R3;

	auto ispec = nnet::data_to_spec(input);
	int size_T = netops::at_spec<typename mpl::pop_front<I_T>::type>(ispec).cols();
	int size_ant = netops::at_spec<typename mpl::pop_front<I_A>::type>(ispec).cols();
	int size_src = netops::at_spec<typename mpl::pop_front<I_L1>::type>(ispec).cols();

	typedef mpl::vector2_c<int,1,0> W_U;
	typedef mpl::vector2_c<int,1,1> W_V;
	typedef mpl::vector2_c<int,1,2> W_antembed;
	typedef mpl::vector2_c<int,1,3> W_srcembed;
	typedef mpl::vector2_c<int,1,4> W_embhid;
	typedef mpl::vector2_c<int,1,5> W_hidout;

	const int size_U = 20;
	const int size_antembed = 20;
	const int size_srcembed = 20;
	const int size_hidden = 50;

	auto wspec = fusion::make_vector(
			lweights(size_T, size_U),
			lweights(size_U, 1),
			lweights(size_ant, size_antembed),
			lweights(size_src, size_srcembed),
			lweights(7 * size_srcembed + size_antembed, size_hidden),
			lweights(size_hidden, NCLASSES));

	typedef nnet::weights<Float,decltype(wspec)> weights;

	auto spec = fusion::make_vector(ispec, wspec);

	using namespace netops;
	auto net = softmax_crossentropy(linear_layer<W_hidout>(spec,
			logistic_sigmoid(linear_layer<W_embhid>(spec,
				logistic_sigmoid(concat(
					linear_layer<W_srcembed>(spec, input_matrix<I_L3>(spec)),
					linear_layer<W_srcembed>(spec, input_matrix<I_L2>(spec)),
					linear_layer<W_srcembed>(spec, input_matrix<I_L1>(spec)),
					linear_layer<W_srcembed>(spec, input_matrix<I_P>(spec)),
					linear_layer<W_srcembed>(spec, input_matrix<I_R1>(spec)),
					linear_layer<W_srcembed>(spec, input_matrix<I_R2>(spec)),
					linear_layer<W_srcembed>(spec, input_matrix<I_R3>(spec)),
					nn6_combiner<I_antmap>(
						linear_layer<W_antembed>(spec, input_matrix<I_A>(spec)),
						logistic_sigmoid(linear_layer<W_V>(spec,
							logistic_sigmoid(linear_layer<W_U>(spec,
								input_matrix<I_T>(spec))))))))))));

	std::cerr << "Net created." << std::endl;

	weights ww(wspec);
	ww.init_normal(.1);
	weights grad(wspec, 0);

	auto inputdata = fusion::make_vector(input, ww.sequence());
	matrix out = net.fprop(inputdata);
	std::cerr << "Forward pass completed." << std::endl;
	net.bprop_loss(targets, inputdata, fusion::make_vector(mpl::vector_c<int,0>(), grad.sequence()));
	std::cerr << "Backward pass completed." << std::endl;

	check<W_hidout,1>(net, input, ww, grad, targets, 0, 2);
	check<W_hidout,1>(net, input, ww, grad, targets, 0, 1);
	check<W_hidout,0>(net, input, ww, grad, targets, 8, 2);
	check<W_hidout,0>(net, input, ww, grad, targets, 1, 1);
	check<W_embhid,1>(net, input, ww, grad, targets, 0, 2);
	check<W_embhid,1>(net, input, ww, grad, targets, 0, 1);
	check<W_embhid,0>(net, input, ww, grad, targets, 6, 2);
	check<W_embhid,0>(net, input, ww, grad, targets, 2, 1);
	check<W_srcembed,1>(net, input, ww, grad, targets, 0, 2);
	check<W_srcembed,1>(net, input, ww, grad, targets, 0, 1);
	check<W_srcembed,0>(net, input, ww, grad, targets, 7, 2);
	check<W_srcembed,0>(net, input, ww, grad, targets, 5, 1);
	check<W_antembed,1>(net, input, ww, grad, targets, 0, 2);
	check<W_antembed,1>(net, input, ww, grad, targets, 0, 0);
	check<W_antembed,0>(net, input, ww, grad, targets, 2, 2);
	check<W_antembed,0>(net, input, ww, grad, targets, 3, 0);
	check<W_V,1>(net, input, ww, grad, targets, 0, 0);
	check<W_V,0>(net, input, ww, grad, targets, 5, 0);
	check<W_V,0>(net, input, ww, grad, targets, 2, 0);
	check<W_U,1>(net, input, ww, grad, targets, 0, 4);
	check<W_U,1>(net, input, ww, grad, targets, 0, 3);
	check<W_U,0>(net, input, ww, grad, targets, 2, 4);
	check<W_U,0>(net, input, ww, grad, targets, 3, 3);

	return 0;
}

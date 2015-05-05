#undef _GNU_SOURCE
#define _GNU_SOURCE
#include <fenv.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

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
typedef unsigned long VocID;

template<class Idx,int WB,class Net,class Input,class Weights>
void check(const Net &net, const Input &input, const Weights &ww, const Weights &grad, const matrix &targets, int i, int j) {
	const Float EPS = 1e-4;

	typedef mpl::push_back<mpl::pop_front<Idx>::type,mpl::int_<WB>>::type WIdx;

	Weights disturb(ww);
	Weights xgrad(ww);

	Float ow = netops::detail::at_spec<WIdx,Weights::map_type>()(ww.sequence())(i,j);

	netops::detail::at_spec<WIdx,Weights::map_type>()(disturb.sequence())(i,j) = ow + EPS;
	matrix out1 = net.fprop(fusion::make_vector(input, disturb.sequence()));
	Float j1 = -targets.cwiseProduct(out1.array().log().matrix()).sum();

	netops::detail::at_spec<WIdx,Weights::map_type>() = ow - EPS;
	matrix out2 = net.fprop(fusion::make_vector(input, disturb.sequence()));
	Float j2 = -targets.cwiseProduct(out2.array().log().matrix()).sum();

	Float g = (j1 - j2) / (2 * EPS);

	std::cout << "< ";
	boost::mpl::for_each<WIdx>([] (auto x) { std::cout << decltype(x)::value << ' '; });
	std::cout << ">(" << i << ',' << j << "): " << g << " == " <<
		netops::detail::at_spec<WIdx,Weights::map_type>()(grad.sequence())(i,j) << std::endl;
}

auto lweights(std::size_t rows, std::size_t cols) {
	return fusion::vector2<nnet::mat_size<Float>,nnet::vec_size<Float>>(
		nnet::mat_size<Float>(rows, cols), nnet::vec_size<Float>(cols));
}

int main() {
	feenableexcept(FE_INVALID | FE_DIVBYZERO);

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

	auto ispec = fusion::make_vector(
			fusion::make_vector(
				nnet::mat_size<Float>(nants, size_ant),
				nnet::mat_size<Float>(nants, size_U),
				nnet::mat_size<Float>(nexmpl, 1)),
			fusion::make_vector(
				nnet::mat_size<Float>(nexmpl, size_src),
				nnet::mat_size<Float>(nexmpl, size_src),
				nnet::mat_size<Float>(nexmpl, size_src),
				nnet::mat_size<Float>(nexmpl, size_src),
				nnet::mat_size<Float>(nexmpl, size_src),
				nnet::mat_size<Float>(nexmpl, size_src),
				nnet::mat_size<Float>(nexmpl, size_src)));

	typedef mpl::vector2_c<int,1,0> W_U;
	typedef mpl::vector2_c<int,1,1> W_V;
	typedef mpl::vector2_c<int,1,2> W_antembed;
	typedef mpl::vector2_c<int,1,3> W_srcembed;
	typedef mpl::vector2_c<int,1,4> W_embhid;
	typedef mpl::vector2_c<int,1,5> W_hidout;

	auto wspec = fusion::make_vector(
			lweights(size_U, size_V),
			lweights(size_V, 1),
			lweights(size_ant, size_antembed),
			lweights(size_src, size_srcembed),
			lweights(7 * size_srcembed + size_antembed, size_hidden),
			lweights(size_hidden, size_out));

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
					nn6_combiner<I_antmap>(spec,
						linear_layer<W_antembed>(spec, input_matrix<I_A>(spec)),
						logistic_sigmoid(linear_layer<W_V>(spec,
							logistic_sigmoid(linear_layer<W_U>(spec,
								input_matrix<I_T>(spec))))))))))));

	weights ww(wspec);
	ww.init_normal(.1);
	weights grad(wspec, 0);

	auto input = load_nn6("/work/users/chm/ncv9.nn6");
	auto inputdata = fusion::make_vector(input, ww.sequence());
	matrix out = net.fprop(inputdata);
	net.bprop_loss(targets, inputdata, fusion::make_vector(mpl::vector_c<int,0>(), grad.sequence()));

	check<W_U,0>(net, input, ww, grad, targets, 3, 3);
	check<W_U,0>(net, input, ww, grad, targets, 2, 4);
	check<W_U,1>(net, input, ww, grad, targets, 0, 3);
	check<W_U,1>(net, input, ww, grad, targets, 0, 4);
	check<W_V,0>(net, input, ww, grad, targets, 2, 3);
	check<W_V,0>(net, input, ww, grad, targets, 5, 6);
	check<W_V,1>(net, input, ww, grad, targets, 0, 3);
	check<W_V,1>(net, input, ww, grad, targets, 0, 6);
	check<W_antembed,0>(net, input, ww, grad, targets, 3, 0);
	check<W_antembed,0>(net, input, ww, grad, targets, 2, 2);
	check<W_antembed,1>(net, input, ww, grad, targets, 0, 0);
	check<W_antembed,1>(net, input, ww, grad, targets, 0, 2);
	check<W_srcembed,0>(net, input, ww, grad, targets, 5, 1);
	check<W_srcembed,0>(net, input, ww, grad, targets, 7, 2);
	check<W_srcembed,1>(net, input, ww, grad, targets, 0, 1);
	check<W_srcembed,1>(net, input, ww, grad, targets, 0, 2);
	check<W_embhid,0>(net, input, ww, grad, targets, 2, 1);
	check<W_embhid,0>(net, input, ww, grad, targets, 6, 2);
	check<W_embhid,1>(net, input, ww, grad, targets, 0, 1);
	check<W_embhid,1>(net, input, ww, grad, targets, 0, 2);
	check<W_hidout,0>(net, input, ww, grad, targets, 1, 1);
	check<W_hidout,0>(net, input, ww, grad, targets, 8, 2);
	check<W_hidout,1>(net, input, ww, grad, targets, 0, 1);
	check<W_hidout,1>(net, input, ww, grad, targets, 0, 2);

	return 0;
}

VocID voc_lookup(const std::string &word, vocmap &voc, bool extend) {
	VocID id;
	vocmap::map_type::const_iterator it = voc.map.find(word);
	if(word != voc.map.end()) 
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

auto load_nn6(const std::string &file, vocmap_type &srcvocmap, vocmap_type &antvocmap) {
	bool make_vocabulary = srcvocmap.empty();

	std::ifstream is(file);
	if(!is.good())
		throw 0;

	std::size_t nexmpl = 0;
	std::size_t nant = 0;

	std::vector<std::string> nn6_lines;
	for(std::string line; getline(is, line);) {
		if(line.compare(0, std::string::npos, "ANAPHORA", 8) == 0)
			nexmpl++;
		else if(line.compare(0, std::string::npos, "ANTECEDENT", 10) == 0)
			nant++;
		nn6_lines.push_back(line);
	}

	typedef Eigen::Matrix<Float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> matrix;
	typedef Eigen::Matrix<Float,Eigen::Dynamic,1,Eigen::RowMajor> vector;
	typedef Eigen::Matrix<VocID,Eigen::Dynamic,1,Eigen::RowMajor> vocid_vector;
	typedef Eigen::Matrix<int,Eigen::Dynamic,1,Eigen::RowMajor> int_vector;

	typedef std::map<std::string,int> classmap_type;
	classmap_type classmap;

	enum { CE, CELA, ELLE, ELLES, IL, ILS, ON, CA, NCLASSES };
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

	matrix targets(nexmpl, NCLASSES);
	vector nada(nexmpl);
	matrix T(nant, nlink);
	int_vector antmap(nexmpl);
	std::vector<Eigen::Triplet<Float>> ant_triplets;
	std::vector<std::vector<Eigen::Triplet<Float>>> srcctx_triplets(7);

	nada.setZero();
	T.setZero();
	antmap.setZero();
	targets.setZero();

	for(std::size_t i = 0, ex = std::numeric_limits<std::size_t>::max(), ant; i < nn6_lines.size(); i++) {
		std::istringstream ss(nn6_lines[i]);
		std::string tag;
		getline(ss, tag, ' ');
		if(tag == "ANAPHORA") {
			if(ex < nexmpl)
				antmap(ex) = ant;

			ex++;
			std::string word;
			for(int j = 0; j < 7; j++) {
				getline(ss, word, ' ');
				VocID v = voc_lookup(word, srcvocmap, make_vocabulary);
				srcctx_triplets[j].push_back(Eigen::Triplet<Float>(ex, v, 1));
			}
			ant = std::numeric_limits<std::size_t>::max();
		} else if(tag == "TARGET") {
			std::string word;
			while(getline(ss, word, ' ')) {
				boost::to_lower(word);
				classmap_type::const_iterator it = classmap.find(word);
				if(it != classmap.end())
					targets(ex, it->second)++;
			}
		} else if(tag == "NADA")
			ss >> nada(ex);
		else if(tag == "ANTECEDENT") {
			ant++;
			int nwords = std::count(nn6_lines[i].begin(), nn6_lines[i].end(), ' ');
			std::string word;
			while(getline(ss, word, ' ')) {
				VocID v = voc_lookup(word, antvocmap, make_vocabulary);
				ant_triplets.push_back(Eigen::Triplet<Float>(ant, v, 1.0 / nwords));
			}
		} else if(tag == "-1") {
			int fidx;
			char colon;
			Float fval;
			while(ss >> fidx >> colon >> fval)
				T(ant, fidx) = fval;
		}
	}

	Eigen::SparseMatrix<Float,Eigen::RowMajor> A(nant, antvocmap.size());
	A.setFromTriplets(ant_triplets.begin(), ant_triplets.end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> L3(nexmpl, srcvocmap.size());
	L3.setFromTriplets(srcctx_triplets[0].begin(), srcctx_triplets[0].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> L2(nexmpl, srcvocmap.size());
	L2.setFromTriplets(srcctx_triplets[1].begin(), srcctx_triplets[1].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> L1(nexmpl, srcvocmap.size());
	L1.setFromTriplets(srcctx_triplets[2].begin(), srcctx_triplets[2].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> P(nexmpl, srcvocmap.size());
	P.setFromTriplets(srcctx_triplets[3].begin(), srcctx_triplets[3].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> R1(nexmpl, srcvocmap.size());
	R1.setFromTriplets(srcctx_triplets[4].begin(), srcctx_triplets[4].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> R2(nexmpl, srcvocmap.size());
	R2.setFromTriplets(srcctx_triplets[5].begin(), srcctx_triplets[5].end());

	Eigen::SparseMatrix<Float,Eigen::RowMajor> R3(nexmpl, srcvocmap.size());
	R3.setFromTriplets(srcctx_triplets[6].begin(), srcctx_triplets[6].end());


	std::size_t srcs = srcvocmap.size();
	return fusion::make_vector(fusion::make_vector(A, T, antmap),
		fusion::make_vector(L3, L2, L1, P, R1, R2, R3));
}

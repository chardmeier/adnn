#ifndef NNET_NN6_H
#define NNET_NN6_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/array.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/back.hpp>
#include <boost/fusion/include/boost_array.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/vector_c.hpp>

#include "nnet.h"
#include "netops.h"

namespace nn6 {

namespace fusion = boost::fusion;
namespace mpl = boost::mpl;

typedef unsigned long voc_id;

namespace detail {

template<class Dataset> class batch_iterator;

} // namespace detail

template<class InputSeq,class Targets>
class nn6_dataset {
public:
	typedef InputSeq input_seq;
	typedef detail::batch_iterator<nn6_dataset<InputSeq,Targets>> batch_iterator;

/*
	typedef fusion::vector2<
			fusion::vector3<sparse_matrix,matrix,int_vector>,
			boost::array<sparse_matrix,7>> input_seq;
*/

	nn6_dataset(input_seq &&input, Targets &&targets) :
		input_(std::forward<InputSeq>(input)), targets_(std::forward<Targets>(targets)) {}

	const input_seq &sequence() const {
		return input_;
	}

	auto spec() const {
		return nnet::data_to_spec(input_);
	}

	std::size_t nitems() const {
		return targets_.rows();
	}

	auto subset(std::size_t from, std::size_t to) const;

	const input_seq &input() const {
		return input_;
	}

	const Targets &targets() const {
		return targets_;
	}

	batch_iterator batch_begin(std::size_t batchsize) const;
	batch_iterator batch_end() const;

private:
	input_seq input_;
	Targets targets_;
};

namespace detail {

template<class Dataset>
using iterator_value_type =
	decltype(std::declval<Dataset>().subset(std::size_t(0),std::size_t(0)));

template<class Dataset>
class batch_iterator
	: public boost::iterator_facade<
			batch_iterator<Dataset>,
			iterator_value_type<Dataset>,
			boost::forward_traversal_tag,
			iterator_value_type<Dataset>> {
public:
	typedef Dataset dataset;

	batch_iterator(const dataset &data, std::size_t batchsize) :
		data_(data), batchsize_(batchsize), pos_(0) {}

	// end iterator
	batch_iterator(const dataset &data) :
		data_(data), batchsize_(0), pos_(data.nitems()) {}

private:
	friend class boost::iterator_core_access;

	const Dataset &data_;
	std::size_t batchsize_;
	std::size_t pos_;

	void increment() {
		pos_ += batchsize_;
		// make sure all end iterators compare equal
		if(pos_ > data_.nitems())
			pos_ = data_.nitems();
	}

	bool equal(const batch_iterator<Dataset> &other) const {
		if(pos_ == other.pos_)
			return true;

		return false;
	}

	const iterator_value_type<Dataset> dereference() const {
		return data_.subset(pos_, pos_ + batchsize_);
	}
};

} // namespace detail

template<class InputSeq,class Targets>
typename nn6_dataset<InputSeq,Targets>::batch_iterator
nn6_dataset<InputSeq,Targets>::batch_begin(std::size_t batchsize) const {
	return batch_iterator(*this, batchsize);
}

template<class InputSeq,class Targets>
typename nn6_dataset<InputSeq,Targets>::batch_iterator
nn6_dataset<InputSeq,Targets>::batch_end() const {
	return batch_iterator(*this);
}

template<class InputSeq,class Targets>
auto make_nn6_dataset(InputSeq &&input, Targets &&targets) {
	return nn6_dataset<InputSeq,Targets>(std::forward<InputSeq>(input), std::forward<Targets>(targets));
}

namespace idx {
	typedef mpl::vector2_c<int,0,0> I_A;
	typedef mpl::vector2_c<int,0,1> I_T;
	typedef mpl::vector2_c<int,0,2> I_antmap;

	typedef mpl::vector2_c<int,1,0> I_L3;
	typedef mpl::vector2_c<int,1,1> I_L2;
	typedef mpl::vector2_c<int,1,2> I_L1;
	typedef mpl::vector2_c<int,1,3> I_P;
	typedef mpl::vector2_c<int,1,4> I_R1;
	typedef mpl::vector2_c<int,1,5> I_R2;
	typedef mpl::vector2_c<int,1,6> I_R3;

	typedef mpl::vector1_c<int,0> W_U;
	typedef mpl::vector1_c<int,1> W_V;
	typedef mpl::vector1_c<int,2> W_antembed;
	typedef mpl::vector1_c<int,3> W_srcembed;
	typedef mpl::vector1_c<int,4> W_embhid;
	typedef mpl::vector1_c<int,5> W_hidout;
} // namespace idx

template<class Float,class Spec,class Net>
class nn6 {
private:
	typedef Spec spec_type;
	spec_type spec_;
	netops::derived_ptr<Net> net_;

public:
	typedef Float float_type;
	typedef nnet::weights<Float,Spec> weight_type;

	nn6(Spec &&spec, netops::expression_ptr<netops::derived_ptr<Net>> &&net) :
		spec_(std::move(spec)), net_(std::move(net).transfer_cast()) {}

	auto spec() const {
		return spec_;
	}

	template<class InputType>
	auto operator()(const weight_type &weights, const InputType &inputs);

	template<class InputType,class OutputType>
	void bprop(const InputType &input, const OutputType &targets, const weight_type &weights, weight_type &grads);

	template<class OutputType,class TargetType>
	float_type error(const OutputType &output, const TargetType &targets) const {
		return -(targets.array() * output.array().log()).sum();
	}
};

enum { CE, CELA, ELLE, ELLES, IL, ILS, ON, CA, OTHER, NCLASSES };

struct vocmap {
	typedef std::unordered_map<std::string,voc_id> map_type;

	voc_id maxid;
	map_type map;

	vocmap() : maxid(1) {
		map.insert(std::make_pair("<unk>", 0));
	}
};

voc_id voc_lookup(const std::string &word, vocmap &voc, bool extend = false) {
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

template<class Float>
auto load_nn6(const std::string &file, vocmap &srcvocmap, vocmap &antvocmap) {
	typedef Eigen::SparseMatrix<Float,Eigen::RowMajor> sparse_matrix;
	typedef Eigen::Matrix<Float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> matrix;
	typedef Eigen::Matrix<int,Eigen::Dynamic,1> int_vector;

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
			ss >> idx; // get rid of -1 at the beginning
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

	typedef Eigen::SparseMatrix<Float,Eigen::RowMajor> wordinput_type;

	wordinput_type A(nant, antvocmap.map.size());
	A.setFromTriplets(ant_triplets.begin(), ant_triplets.end());

	wordinput_type L3(nexmpl, srcvocmap.map.size());
	L3.setFromTriplets(srcctx_triplets[0].begin(), srcctx_triplets[0].end());

	wordinput_type L2(nexmpl, srcvocmap.map.size());
	L2.setFromTriplets(srcctx_triplets[1].begin(), srcctx_triplets[1].end());

	wordinput_type L1(nexmpl, srcvocmap.map.size());
	L1.setFromTriplets(srcctx_triplets[2].begin(), srcctx_triplets[2].end());

	wordinput_type P(nexmpl, srcvocmap.map.size());
	P.setFromTriplets(srcctx_triplets[3].begin(), srcctx_triplets[3].end());

	wordinput_type R1(nexmpl, srcvocmap.map.size());
	R1.setFromTriplets(srcctx_triplets[4].begin(), srcctx_triplets[4].end());

	wordinput_type R2(nexmpl, srcvocmap.map.size());
	R2.setFromTriplets(srcctx_triplets[5].begin(), srcctx_triplets[5].end());

	wordinput_type R3(nexmpl, srcvocmap.map.size());
	R3.setFromTriplets(srcctx_triplets[6].begin(), srcctx_triplets[6].end());

	return make_nn6_dataset(
		fusion::make_vector(fusion::make_vector(A, T, antmap),
			boost::array<wordinput_type,7>({ L3, L2, L1, P, R1, R2, R3 })),
		std::move(targets));
}

template<class Float>
auto lweights(std::size_t rows, std::size_t cols) {
	return fusion::vector2<nnet::mat_size<Float>,nnet::vec_size<Float>>(
		nnet::mat_size<Float>(rows, cols), nnet::vec_size<Float>(cols));
}

template<class Float,class Inputs>
auto make_nn6(const Inputs &input,
		std::size_t size_U, std::size_t size_antembed, std::size_t size_srcembed, std::size_t size_hidden) {
	auto ispec = nnet::data_to_spec(input);
	int size_T = netops::at_spec<idx::I_T>(ispec).cols();
	int size_ant = netops::at_spec<idx::I_A>(ispec).cols();
	int size_src = netops::at_spec<idx::I_L1>(ispec).cols();

	auto wspec = fusion::make_vector(
			lweights<Float>(size_T, size_U),
			lweights<Float>(size_U, 1),
			lweights<Float>(size_ant, size_antembed),
			lweights<Float>(size_src, size_srcembed),
			lweights<Float>(7 * size_srcembed + size_antembed, size_hidden),
			lweights<Float>(size_hidden, NCLASSES));

	typedef nnet::weights<Float,decltype(wspec)> weights;

	using namespace netops;
	auto &&net = softmax_crossentropy(linear_layer<idx::W_hidout>(wspec,
			logistic_sigmoid(linear_layer<idx::W_embhid>(wspec,
				logistic_sigmoid(concat(
					linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_L3>(ispec)),
					linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_L2>(ispec)),
					linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_L1>(ispec)),
					linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_P>(ispec)),
					linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_R1>(ispec)),
					linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_R2>(ispec)),
					linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_R3>(ispec)),
					nn6_combiner<idx::I_antmap>(
						linear_layer<idx::W_antembed>(wspec, input_matrix<idx::I_A>(ispec)),
						logistic_sigmoid(linear_layer<idx::W_V>(wspec,
							logistic_sigmoid(linear_layer<idx::W_U>(wspec,
								input_matrix<idx::I_T>(ispec))))))))))));

	typedef typename std::remove_reference<decltype(net)>::type::expr_type net_type;
	return nn6<Float,decltype(wspec),net_type>(std::move(wspec), std::move(net));
}

template<class Float,class Net,class Spec>
template<class InputType>
auto nn6<Float,Net,Spec>::operator()(const weight_type &weights, const InputType &input) {
	return net_.fprop(input, weights.sequence());
}

template<class Float,class Net,class Spec>
template<class InputType,class OutputType>
void nn6<Float,Net,Spec>::bprop(const InputType &input, const OutputType &targets, const weight_type &weights, weight_type &grads) {
	net_.bprop_loss(targets, input, weights.sequence(), grads.sequence());
}

template<class InputSeq,class Targets>
auto nn6_dataset<InputSeq,Targets>::subset(std::size_t from, std::size_t to) const {
	const auto &A = netops::at_spec<idx::I_A>(input_);
	const auto &T = netops::at_spec<idx::I_T>(input_);
	const auto &antmap = netops::at_spec<idx::I_antmap>(input_);
	const auto &srcctx = fusion::back(input_);
	int nexmpl = std::min(to, nitems()) - from;
	int from_ant = antmap.head(from).sum();
	int n_ant = antmap.middleRows(from, nexmpl).sum();
	return make_nn6_dataset(
		fusion::make_vector(
			fusion::make_vector(A.middleRows(from_ant, n_ant).eval(), T.middleRows(from_ant, n_ant).eval(),
				antmap.middleRows(from, nexmpl).eval()),
			fusion::as_vector(fusion::transform(srcctx, [from, nexmpl] (const auto &m) { return m.middleRows(from, nexmpl).eval(); }))),
		targets_.middleRows(from, nexmpl).eval());
}

} // namespace nn6

#endif

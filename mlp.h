#ifndef NNET_MLP_H
#define NNET_MLP_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/array.hpp>
#include <boost/fusion/include/back.hpp>
#include <boost/fusion/include/boost_array.hpp>
#include <boost/fusion/include/front.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/vector_c.hpp>

#include "nnet.h"
#include "netops.h"

namespace mlp {

namespace fusion = boost::fusion;
namespace mpl = boost::mpl;

namespace idx {
	typedef mpl::vector2_c<int,0,0> I1;

	typedef mpl::vector2_c<int,0,0> W1;
	typedef mpl::vector2_c<int,0,1> W2;
} // namespace idx

template<class Float,class Spec,class Net>
class mlp {
private:
	typedef Spec spec_type;
	spec_type spec_;
	netops::derived_ptr<Net> net_;

public:
	typedef Float float_type;
	typedef nnet::weights<Float,Spec> weight_type;

	mlp(Spec &&spec, netops::expression_ptr<netops::derived_ptr<Net>> &&net) :
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

template<class Float>
auto lweights(std::size_t rows, std::size_t cols) {
	return fusion::vector2<nnet::mat_size<Float>,nnet::vec_size<Float>>(
		nnet::mat_size<Float>(rows, cols), nnet::vec_size<Float>(cols));
}

template<class Float,class Inputs>
auto make_mlp(const Inputs &input, std::size_t size_hidden) {
	auto ispec = nnet::data_to_spec(input);
	int size_I1 = netops::at_spec<idx::I1>(ispec).cols();

	auto wspec = fusion::make_vector(
		fusion::make_vector(
			lweights<Float>(size_I1, size_hidden),
			lweights<Float>(size_hidden, 3)));

	typedef nnet::weights<Float,decltype(wspec)> weights;

	using namespace netops;
	auto &&net = softmax_crossentropy(linear_layer<idx::W2>(wspec,
				logistic_sigmoid(linear_layer<idx::W1>(wspec, input_matrix<idx::I1>(ispec)))));

	typedef typename std::remove_reference<decltype(net)>::type::expr_type net_type;
	return mlp<Float,decltype(wspec),net_type>(std::move(wspec), std::move(net));
}

template<class Float,class Net,class Spec>
template<class InputType>
auto mlp<Float,Net,Spec>::operator()(const weight_type &weights, const InputType &input) {
	return net_.fprop(input, weights.sequence());
}

template<class Float,class Net,class Spec>
template<class InputType,class OutputType>
void mlp<Float,Net,Spec>::bprop(const InputType &input, const OutputType &targets, const weight_type &weights, weight_type &grads) {
	net_.bprop_loss(targets, input, weights.sequence(), grads.sequence());
}

template<class Float>
class mlp_batch_iterator;

template<class Float>
class mlp_dataset {
private:
	typedef nnet::std_matrix<Float> input_type;
	typedef nnet::std_matrix<Float> output_type;

	fusion::vector1<boost::array<input_type,1>> inputseq_;
	input_type &inputs_;
	output_type targets_;

public:
	typedef mlp_batch_iterator<Float> batch_iterator;

	struct input_transformation {
		nnet::std_array<typename input_type::Scalar,1,Eigen::Dynamic> mean;
		nnet::std_array<typename input_type::Scalar,1,Eigen::Dynamic> std;
	};

	mlp_dataset() : inputs_(fusion::front(fusion::front(inputseq_))) {}

	mlp_dataset(const input_type &inputs, const output_type &targets) :
		inputseq_(boost::array<input_type,1>({inputs})), inputs_(fusion::front(fusion::front(inputseq_))), targets_(targets) {}

	const input_type &inputs() const {
		return inputs_;
	}

	input_type &inputs() {
		return inputs_;
	}

	const output_type &targets() const {
		return targets_;
	}

	output_type &targets() {
		return targets_;
	}

	std::size_t nitems() const {
		return inputs_.rows();
	}

	auto sequence() const {
		return inputseq_;
	}

	batch_iterator batch_begin(std::size_t batchsize) const;
	batch_iterator batch_end() const;

	void shuffle();
	auto subset(std::size_t from, std::size_t to);
	auto subset(std::size_t from, std::size_t to) const;
	input_transformation shift_scale();

	//template<class InputMatrix2,class OutputMatrix2>
	//void transform_input(const typename mlp_dataset<Net,InputMatrix2,OutputMatrix2>::input_transformation &s);
	void transform_input(const input_transformation &s);
};

template<class InputMatrix,class OutputMatrix>
mlp_dataset<typename InputMatrix::Scalar>
make_mlp_dataset(const InputMatrix &inputs, const OutputMatrix &targets) {
	return mlp_dataset<typename InputMatrix::Scalar>(inputs, targets);
}

template<class Float>
struct facade {
	typedef decltype(std::declval<mlp_dataset<Float>>().subset(std::size_t(0),std::size_t(0))) value_type;
	typedef boost::iterator_facade<mlp_batch_iterator<Float>,
				value_type, boost::forward_traversal_tag, value_type>
		type;
};

template<class Float>
class mlp_batch_iterator
	: public facade<Float>::type {
public:
	typedef mlp_dataset<Float> dataset;

	mlp_batch_iterator(const dataset &data, std::size_t batchsize) :
		data_(data), batchsize_(batchsize), pos_(0) {}

	// end iterator
	mlp_batch_iterator(const dataset &data) :
		data_(data), batchsize_(0), pos_(data.nitems()) {}

private:
	friend class boost::iterator_core_access;

	using typename facade<Float>::type::value_type;

	const mlp_dataset<Float> &data_;
	std::size_t batchsize_;
	std::size_t pos_;

	void increment() {
		pos_ += batchsize_;
		// make sure all end iterators compare equal
		if(pos_ > data_.nitems())
			pos_ = data_.nitems();
	}

	bool equal(const mlp_batch_iterator &other) const {
		if(pos_ == other.pos_)
			return true;

		return false;
	}

	const value_type dereference() const {
		return data_.subset(pos_, pos_ + batchsize_);
	}
};

template<class Float>
typename mlp_dataset<Float>::batch_iterator
mlp_dataset<Float>::batch_begin(std::size_t batchsize) const {
	return batch_iterator(*this, batchsize);
}

template<class Float>
typename mlp_dataset<Float>::batch_iterator
mlp_dataset<Float>::batch_end() const {
	return batch_iterator(*this);
}

template<class Float>
void mlp_dataset<Float>::shuffle() {
	Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(nitems());
	perm.setIdentity();
	std::random_device rd;
	std::mt19937 rgen(rd());
	std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), rgen);
	inputs_.matrix() = perm * inputs_.matrix();
	targets_.matrix() = perm * targets_.matrix();
}

template<class Float>
typename mlp_dataset<Float>::input_transformation mlp_dataset<Float>::shift_scale() {
	input_transformation t;
	t.mean = inputs_.matrix().colwise().mean();
	inputs_.matrix() = inputs_.array().rowwise() - t.mean;
	t.std = inputs_.matrix().cwiseProduct(inputs_.matrix()).colwise().mean().cwiseSqrt();
	inputs_.matrix() = inputs_.array().rowwise() / t.std;
	return t;
}

template<class Float>
void mlp_dataset<Float>::transform_input(const input_transformation &s) {
	inputs_.matrix() = (inputs_.array().rowwise() - s.mean).rowwise() / s.std;
}

template<class Float>
auto mlp_dataset<Float>::subset(std::size_t from, std::size_t to) {
	to = std::min(to, nitems());
	return make_mlp_dataset(inputs_.matrix().middleRows(from, to - from), targets_.matrix().middleRows(from, to - from));
}

template<class Float>
auto mlp_dataset<Float>::subset(std::size_t from, std::size_t to) const {
	to = std::min(to, nitems());
	return make_mlp_dataset(inputs_.matrix().middleRows(from, to - from), targets_.matrix().middleRows(from, to - from));
}

} // namespace mlp

#endif

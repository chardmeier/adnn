#ifndef NNET_MLP_H
#define NNET_MLP_H

#include <algorithm>
#include <random>
#include <type_traits>

#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "nnet.h"

namespace nnet {

template<class Net,class InputMatrix,class OutputMatrix>
class mlp_dataset;

template<class ActivationVector,class F>
class mlp {
private:
	typedef boost::fusion::vector<mat_size,mat_size> spec_type;
	spec_type spec_;

public:
	typedef F float_type;

	typedef std_matrix<float_type> float_matrix;
	typedef mlp_dataset<mlp<ActivationVector,F>,float_matrix,float_matrix> dataset;

	template<class Matrix>
	class input_type {
	private:
		Matrix matrix_;

	public:
		typedef typename Matrix::Scalar float_type;

		input_type() {}
		input_type(const Matrix &data) : matrix_(data) {}

		Matrix &matrix() {
			return matrix_;
		}

		const Matrix &matrix() const {
			return matrix_;
		}

		auto array() {
			return matrix_.array();
		}

		auto array() const {
			return matrix_.array();
		}

		std::size_t nitems() const {
			return matrix_.rows();
		}

		template<class NewType>
		auto cast() const {
			return matrix_.cast<NewType>();
		}
	};

	template<class FF>
	using basic_input_type = input_type<std_matrix<FF> >;

	template<class Matrix>
	class output_type {
	private:
		Matrix matrix_;

	public:
		typedef typename Matrix::Scalar float_type;

		output_type() {}
		output_type(const Matrix &data) : matrix_(data) {}

		Matrix &matrix() {
			return matrix_;
		}

		const Matrix &matrix() const {
			return matrix_;
		}

		auto array() {
			return matrix_.array();
		}

		auto array() const {
			return matrix_.array();
		}

		std::size_t nitems() const {
			return matrix_.size();
		}

		template<class NewType>
		auto cast() const {
			return matrix_.cast<NewType>();
		}
	};

	template<class FF>
	using basic_output_type = output_type<std_matrix<FF> >;

	template<class FF>
	using weight_type = weights<FF,spec_type,std_array<FF> >;
	
	mlp(size_t input, size_t hidden, size_t output) :
		spec_(mat_size(input, hidden), mat_size(hidden, output)) {}

	template<class FF,class InputMatrix>
	auto operator()(const weight_type<FF> &W, const input_type<InputMatrix> &inp) const;

	const spec_type &spec() const {
		return spec_;
	}
};

template<class ActivationVector,class A>
template<class FF,class InputMatrix>
auto mlp<ActivationVector,A>::operator()(const weight_type<FF> &w, const input_type<InputMatrix> &inp) const {
	const auto &p1 = (inp.matrix() * w.template at<0>()).eval();
	typedef typename boost::fusion::result_of::value_at_c<ActivationVector,0>::type::template functor<decltype(p1)> Activation1;
	const auto &a1 = Activation1()(p1).eval();
	const auto &p2 = (a1 * w.template at<1>()).eval();
	typedef typename boost::fusion::result_of::value_at_c<ActivationVector,1>::type::template functor<decltype(p2)> Activation2;
	std_matrix<FF> out = Activation2()(p2).eval();
	return output_type<std_matrix<FF> >(out);
}

template<class Net,class InputMatrix,class OutputMatrix>
class mlp_batch_iterator;

template<class Net,class InputMatrix,class OutputMatrix>
class mlp_dataset {
private:
	typedef Net net_type;

	typedef typename net_type::template input_type<InputMatrix> input_type;
	typedef typename net_type::template output_type<OutputMatrix> output_type;

	input_type inputs_;
	output_type targets_;
public:
	typedef mlp_batch_iterator<Net,InputMatrix,OutputMatrix> batch_iterator;

	struct input_transformation {
		std_array<typename InputMatrix::Scalar,1,Eigen::Dynamic> mean;
		std_array<typename InputMatrix::Scalar,1,Eigen::Dynamic> std;
	};

	mlp_dataset() {}

	mlp_dataset(const InputMatrix &inputs, const OutputMatrix &targets) :
		inputs_(inputs), targets_(targets) {}

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
		return inputs_.nitems();
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

template<class Net,class InputMatrix,class OutputMatrix>
mlp_dataset<Net,InputMatrix,OutputMatrix>
make_mlp_dataset(InputMatrix inputs, OutputMatrix targets) {
	return mlp_dataset<Net,InputMatrix,OutputMatrix>(inputs, targets);
}

template<class Net,class InputMatrix,class OutputMatrix>
struct facade {
	typedef decltype(DECLVAL(mlp_dataset<Net,const InputMatrix,const OutputMatrix> )().subset(std::size_t(0),std::size_t(0))) value_type;
	typedef boost::iterator_facade<mlp_batch_iterator<Net,InputMatrix,OutputMatrix>,
				value_type, boost::forward_traversal_tag, value_type>
		type;
};

template<class Net,class InputMatrix,class OutputMatrix>
class mlp_batch_iterator
	: public facade<Net,InputMatrix,OutputMatrix>::type {
public:
	typedef mlp_dataset<Net,InputMatrix,OutputMatrix> dataset;

	mlp_batch_iterator(const dataset &data, std::size_t batchsize) :
		data_(data), batchsize_(batchsize), pos_(0) {}

	// end iterator
	mlp_batch_iterator(const dataset &data) :
		data_(data), batchsize_(0), pos_(data.nitems()) {}

private:
	friend class boost::iterator_core_access;

	using typename facade<Net,InputMatrix,OutputMatrix>::type::value_type;

	const mlp_dataset<Net,InputMatrix,OutputMatrix> &data_;
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

template<class Net,class InputMatrix,class OutputMatrix>
typename mlp_dataset<Net,InputMatrix,OutputMatrix>::batch_iterator
mlp_dataset<Net,InputMatrix,OutputMatrix>::batch_begin(std::size_t batchsize) const {
	return batch_iterator(*this, batchsize);
}

template<class Net,class InputMatrix,class OutputMatrix>
typename mlp_dataset<Net,InputMatrix,OutputMatrix>::batch_iterator
mlp_dataset<Net,InputMatrix,OutputMatrix>::batch_end() const {
	return batch_iterator(*this);
}

template<class Net,class InputMatrix,class OutputMatrix>
void mlp_dataset<Net,InputMatrix,OutputMatrix>::shuffle() {
	Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(nitems());
	perm.setIdentity();
	std::random_device rd;
	std::mt19937 rgen(rd());
	std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), rgen);
	inputs_.matrix() = perm * inputs_.matrix();
	targets_.matrix() = perm * targets_.matrix();
}

template<class Net,class InputMatrix,class OutputMatrix>
typename mlp_dataset<Net,InputMatrix,OutputMatrix>::input_transformation mlp_dataset<Net,InputMatrix,OutputMatrix>::shift_scale() {
	input_transformation t;
	t.mean = inputs_.matrix().colwise().mean();
	inputs_.matrix() = inputs_.array().rowwise() - t.mean;
	t.std = inputs_.matrix().cwiseProduct(inputs_.matrix()).colwise().mean().cwiseSqrt();
	inputs_.matrix() = inputs_.array().rowwise() / t.std;
	return t;
}

//template<class Net,class InputMatrix,class OutputMatrix>
//template<class InputMatrix2,class OutputMatrix2>
//void mlp_dataset<Net,InputMatrix,OutputMatrix>::transform_input(const typename mlp_dataset<Net,InputMatrix2,OutputMatrix2>::input_transformation &s) {
	//inputs_.matrix() = (inputs_.matrix().rowwise() - s.mean).cwiseQuotient(s.std);
//}

template<class Net,class InputMatrix,class OutputMatrix>
void mlp_dataset<Net,InputMatrix,OutputMatrix>::transform_input(const input_transformation &s) {
	inputs_.matrix() = (inputs_.array().rowwise() - s.mean).rowwise() / s.std;
}

template<class Net,class InputMatrix,class OutputMatrix>
auto mlp_dataset<Net,InputMatrix,OutputMatrix>::subset(std::size_t from, std::size_t to) {
	to = std::min(to, nitems());
	return make_mlp_dataset<Net>(inputs_.matrix().middleRows(from, to - from), targets_.matrix().middleRows(from, to - from));
}

template<class Net,class InputMatrix,class OutputMatrix>
auto mlp_dataset<Net,InputMatrix,OutputMatrix>::subset(std::size_t from, std::size_t to) const {
	to = std::min(to, nitems());
	return make_mlp_dataset<Net>(inputs_.matrix().middleRows(from, to - from), targets_.matrix().middleRows(from, to - from));
}

} // namespace nnet

#endif

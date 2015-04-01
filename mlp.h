#ifndef NNET_MLP_H
#define NNET_MLP_H

#include <algorithm>
#include <random>
#include <type_traits>

#include <boost/fusion/include/intrinsic.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "nnet.h"

namespace nnet {

template<class Net,class InputMatrix,class OutputMatrix>
class mlp_dataset;

template<class ActivationVector,class F>
class mlp {
private:
	std::size_t input_, hidden_, output_;

public:
	typedef F float_type;

	typedef Eigen::Matrix<float_type,Eigen::Dynamic,Eigen::Dynamic> float_matrix;
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

		std::size_t nitems() const {
			return matrix_.rows();
		}

		template<class NewType>
		auto cast() const {
			return matrix_.cast<NewType>();
		}
	};

	template<class FF>
	struct basic_input_type {
		typedef input_type<Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic> > type;
	};

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

		std::size_t nitems() const {
			return matrix_.size();
		}

		template<class NewType>
		auto cast() const {
			return matrix_.cast<NewType>();
		}
	};

	template<class FF>
	struct basic_output_type {
		typedef output_type<Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic> > type;
	};

	template<class FF>
	class weight_type {
	private:
		template<typename> friend class weight_type;

		std::size_t inp_, hid_, out_;
		Eigen::Array<FF,Eigen::Dynamic,1> data_;
		Eigen::Map<Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic> > w1_, w2_;

		template<class Derived>
		weight_type(std::size_t inp, std::size_t hid, std::size_t out,
			const Eigen::ArrayBase<Derived> &data) :
				inp_(inp), hid_(hid), out_(out),
				data_(data),
				w1_(data_.data(), inp_, hid_),
				w2_(data_.data() + w1_.size(), hid_, out_) {}

	public:
		typedef FF float_type;

		weight_type(const mlp<ActivationVector,F> &net) :
				inp_(net.input_), hid_(net.hidden_), out_(net.output_),
				data_(inp_ * hid_ + hid_ * out_, 1),
				w1_(data_.data(), inp_, hid_),
				w2_(data_.data() + w1_.size(), hid_, out_) {}

		weight_type(const mlp<ActivationVector,F> &net, const FF &value) :
				inp_(net.input_), hid_(net.hidden_), out_(net.output_),
				data_(inp_ * hid_ + hid_ * out_, 1),
				w1_(data_.data(), inp_, hid_),
				w2_(data_.data() + w1_.size(), hid_, out_) {
			data_.setConstant(value);
		}

		template<class FFF>
		weight_type(const weight_type<FFF> &o) :
				inp_(o.inp_), hid_(o.hid_), out_(o.out_),
				data_(o.data_.template cast<FF>()),
				w1_(data_.data(), inp_, hid_),
				w2_(data_.data() + w1_.size(), hid_, out_) {}

		weight_type(const weight_type<FF> &o) :
				inp_(o.inp_), hid_(o.hid_), out_(o.out_),
				data_(o.data_),
				w1_(data_.data(), inp_, hid_),
				w2_(data_.data() + w1_.size(), hid_, out_) {}

		weight_type<FF> &operator=(const weight_type<FF> &o) {
			inp_ = o.inp_;
			hid_ = o.hid_;
			out_ = o.out_;
			data_ = o.data_;
			new(&w1_) Eigen::Map<Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic> >(data_.data(), inp_, hid_);
			new(&w2_) Eigen::Map<Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic> >(data_.data() + w1_.size(), hid_, out_);
			return *this;
		}

		template<class Functor>
		weight_type<typename Functor::result_type> transform(const Functor &f) const {
			return weight_type<typename Functor::result_type>(w1_.rows(), w1_.cols(), w2_.cols(),
				data_.unaryExpr(f));
		}

		auto &array() {
			return data_;
		}

		const auto &array() const {
			return data_;
		}

		auto &w1() {
			return w1_;
		}

		const auto &w1() const {
			return w1_;
		}

		auto &w2() {
			return w2_;
		}

		const auto &w2() const {
			return w2_;
		}

		void init_normal(float_type stddev) {
			std::random_device rd;
			std::mt19937 rgen(rd());
			std::normal_distribution<float_type> dist(0, stddev);
			std::generate_n(data_.data(), data_.size(), std::bind(dist, rgen));
		}
	};

	mlp(size_t input, size_t hidden, size_t output) :
		input_(input), hidden_(hidden), output_(output) {}

	template<class FF,class InputMatrix>
	auto operator()(const weight_type<FF> &W, const input_type<InputMatrix> &inp) const;
};

template<class ActivationVector,class A>
template<class FF,class InputMatrix>
auto mlp<ActivationVector,A>::operator()(const weight_type<FF> &w, const input_type<InputMatrix> &inp) const {
	const auto &p1 = (inp.matrix() * w.w1()).eval();
	typedef typename boost::fusion::result_of::value_at_c<ActivationVector,0>::type::
		template functor<typename std::remove_const<typename std::remove_reference<decltype(p1)>::type>::type> Activation1;
	const auto &a1 = Activation1()(p1).eval();
	const auto &p2 = (a1 * w.w2()).eval();
	typedef Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic> outmatrix;
	typedef typename boost::fusion::result_of::value_at_c<ActivationVector,1>::type::
		template functor<typename std::remove_const<typename std::remove_reference<decltype(p2)>::type>::type> Activation2;
	outmatrix out = Activation2()(p2).eval();
	return output_type<outmatrix>(out);
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
	auto subset(std::size_t from, std::size_t to) const;
};

template<class Net,class InputMatrix,class OutputMatrix>
mlp_dataset<Net,InputMatrix,OutputMatrix>
make_mlp_dataset(InputMatrix inputs, OutputMatrix targets) {
	return mlp_dataset<Net,InputMatrix,OutputMatrix>(inputs, targets);
}

template<class Net,class InputMatrix,class OutputMatrix>
struct facade {
	typedef decltype(std::declval<mlp_dataset<Net,InputMatrix,OutputMatrix> >().subset(std::size_t(0),std::size_t(0))) value_type;
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
auto mlp_dataset<Net,InputMatrix,OutputMatrix>::subset(std::size_t from, std::size_t to) const {
	to = std::min(to, nitems());
	return make_mlp_dataset<Net>(inputs_.matrix().middleRows(from, to - from), targets_.matrix().middleRows(from, to - from));
}

} // namespace nnet

#endif

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <type_traits>

#include <boost/fusion/include/intrinsic.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <Eigen/Core>
#include <adept.h>

template<class ActivationVector,class A = adept::aReal>
class net_3layer;

template<class A>
struct ad_types;

template<>
struct ad_types<adept::aReal> {
    typedef adept::Real float_type;
    typedef adept::aReal afloat_type;
};

template<>
struct ad_types<adept::Real> {
    typedef adept::Real float_type;
    typedef adept::aReal afloat_type;
};

template<class C>
struct value_functor {
	typedef typename ad_types<C>::float_type result_type;
	result_type operator()(const C &in) const {
		return adept::value(in);
	}
};

// we define our own functors for exp and log to make sure the right functions are found
// both for adept::aReal and for plain floats.
template<class F>
struct exp_functor {
	typedef F return_type;
	F operator()(const F &x) const {
		return exp(x);
	}
};

template<class F>
struct log_functor {
	typedef F return_type;
	F operator()(const F &x) const {
		return log(x);
	}
};

struct sigmoid {
	template<class Derived>
	struct functor {
		auto operator()(const Eigen::MatrixBase<typename std::remove_const<typename std::remove_reference<Derived>::type>::type> &x) const {
			typedef typename Derived::Scalar F;
			return (F(1) / (F(1) + (-x).array().unaryExpr(exp_functor<F>()))).matrix();
		}
		typedef decltype(std::declval<functor<Derived> >()(std::declval<const Eigen::MatrixBase<Derived> >())) result_type;
	};
};

struct softmax {
	template<class Derived>
	struct functor {
		auto operator()(const Eigen::MatrixBase<typename std::remove_const<typename std::remove_reference<Derived>::type>::type> &x) const {
			typedef typename Derived::Scalar F;
			auto a = (x.array() - x.array().rowwise().maxCoeff().replicate(1, x.cols())).unaryExpr(exp_functor<F>());
			return (a / a.rowwise().sum().replicate(1, x.cols())).matrix();
		}
		typedef decltype(std::declval<functor<Derived> >()(std::declval<const Eigen::MatrixBase<Derived> >())) result_type;
	};
};

struct quadratic_loss {
	template<class T1,class T2>
	struct functor {
		typedef typename T1::float_type result_type;
		result_type operator()(const T1 &output, const T2 &targets) const {
			const auto &err = output.matrix() - targets.matrix();
			return err.cwiseProduct(err).sum() / targets.nitems();
		}
	};
};

struct crossentropy_loss {
	template<class T1,class T2>
	struct functor {
		typedef typename T1::float_type result_type;
		result_type operator()(const T1 &output, const T2 &targets) const {
			return -output.matrix().unaryExpr(log_functor<result_type>()).cwiseProduct(targets.matrix()).sum() / targets.nitems();
		}
	};
};

template<class Loss,class T1,class T2>
typename T1::float_type evaluate_loss(const Loss &loss, const T1 &output, const T2 &targets) {
	typename Loss::template functor<T1,T2> lossfn;
	return lossfn(output, targets);
}

template<class N, class Loss>
class net_wrapper {
public:
	typedef typename N::float_type float_type;
	typedef typename N::template basic_input_type<float_type>::type input_type;
	typedef typename N::template basic_output_type<float_type>::type output_type;
	typedef typename N::template weight_type<float_type> weight_type;

private:
	typedef typename N::afloat_type afloat_type;
	typedef typename N::template basic_input_type<afloat_type>::type ainput_type;
	typedef typename N::template basic_output_type<afloat_type>::type aoutput_type;
	typedef typename N::template weight_type<afloat_type> aweight_type;

	typedef N net_type;

	net_type net_;
	Loss loss_;

public:
	net_wrapper(const net_type &net, const Loss &loss) :
		net_(net), loss_(loss) {}

	template<class InputType,class OutputType>
	float_type operator()(const weight_type &W, const InputType &inp, const OutputType &targets, weight_type &grad) const;
};

template<class N, class Loss>
net_wrapper<N,Loss> wrap_net(const N &net, const Loss &loss) {
	typedef typename N::template output_type<typename N::float_type> otype;
	typedef typename N::template output_type<typename N::afloat_type> aotype;
	return net_wrapper<N,Loss>(net, typename Loss::template functor<aotype>());
}

template<class N, class Loss>
template<class InputType,class OutputType>
typename net_wrapper<N,Loss>::float_type net_wrapper<N,Loss>::operator()(const weight_type &W, const InputType &inp,
		const OutputType &targets, weight_type &grad) const {
	adept::Stack stack;
	aweight_type aW(W);
	ainput_type ainp(inp.template cast<afloat_type>());
	aoutput_type atargets(targets.template cast<afloat_type>());
	stack.new_recording();
	aoutput_type aout = net_(aW, ainp);
	afloat_type err = evaluate_loss(loss_, aout, atargets);
	err.set_gradient(float_type(1));
	stack.compute_adjoint();
	grad = aW.get_gradients();
	//std::cerr << "w1 grad:\n" << grad.w1() << std::endl;
	return err.value();
}

template<class Net,class InputMatrix,class OutputMatrix>
class net_3layer_dataset;

template<class ActivationVector,class A>
class net_3layer {
private:
	std::size_t input_, hidden_, output_;

public:
	typedef A afloat_type;
	typedef typename ad_types<A>::float_type float_type;

	typedef Eigen::Matrix<float_type,Eigen::Dynamic,Eigen::Dynamic> float_matrix;
	typedef net_3layer_dataset<net_3layer<ActivationVector,A>,float_matrix,float_matrix> dataset;

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

		weight_type(std::size_t inp, std::size_t hid, std::size_t out) :
				inp_(inp), hid_(hid), out_(out),
				data_(inp_ * hid_ + hid_ * out_, 1),
				w1_(data_.data(), inp_, hid_),
				w2_(data_.data() + w1_.size(), hid_, out_) {}

	public:
		typedef FF float_type;

		weight_type(const net_3layer<ActivationVector,A> &net) :
				inp_(net.input_), hid_(net.hidden_), out_(net.output_),
				data_(inp_ * hid_ + hid_ * out_, 1),
				w1_(data_.data(), inp_, hid_),
				w2_(data_.data() + w1_.size(), hid_, out_) {}

		weight_type(const net_3layer<ActivationVector,A> &net, const FF &value) :
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

                weight_type<typename ad_types<FF>::float_type> get_gradients() {
                    weight_type<typename ad_types<FF>::float_type> out(w1_.rows(), w1_.cols(), w2_.cols());
                    adept::get_gradients(data_.data(), data_.size(), out.data_.data());
                    return out;
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

	net_3layer(size_t input, size_t hidden, size_t output) :
		input_(input), hidden_(hidden), output_(output) {}

	template<class FF,class InputMatrix>
	auto operator()(const weight_type<FF> &W, const input_type<InputMatrix> &inp) const;
};

template<class ActivationVector,class A>
template<class FF,class InputMatrix>
auto net_3layer<ActivationVector,A>::operator()(const weight_type<FF> &w, const input_type<InputMatrix> &inp) const {
	struct val {
		typedef float_type result_type;
		float_type operator()(const afloat_type &v) const {
			return v.value();
		}
	};

	//std::cerr << "w1:\n" << w.w1().unaryExpr(val()) << std::endl;
	const auto &p1 = (inp.matrix() * w.w1()).eval();
	typedef typename boost::fusion::result_of::value_at_c<ActivationVector,0>::type::
		template functor<typename std::remove_const<typename std::remove_reference<decltype(p1)>::type>::type> Activation1;
	//std::cerr << "p1:\n" << p1.unaryExpr(val()) << std::endl;
	const auto &a1 = Activation1()(p1).eval();
	const auto &p2 = (a1 * w.w2()).eval();
	typedef Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic> outmatrix;
	typedef typename boost::fusion::result_of::value_at_c<ActivationVector,1>::type::
		template functor<typename std::remove_const<typename std::remove_reference<decltype(p2)>::type>::type> Activation2;
	outmatrix out = Activation2()(p2).eval();
	//std::cerr << "out:\n" << out.unaryExpr(val()) << std::endl;
	return output_type<outmatrix>(out);
}

template<class Net,class InputMatrix,class OutputMatrix>
class net_3layer_batch_iterator;

template<class Net,class InputMatrix,class OutputMatrix>
class net_3layer_dataset {
private:
	typedef Net net_type;

	typedef typename net_type::template input_type<InputMatrix> input_type;
	typedef typename net_type::template output_type<OutputMatrix> output_type;

	input_type inputs_;
	output_type targets_;
public:
	typedef net_3layer_batch_iterator<Net,InputMatrix,OutputMatrix> batch_iterator;

	net_3layer_dataset() {}

	net_3layer_dataset(const InputMatrix &inputs, const OutputMatrix &targets) :
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
net_3layer_dataset<Net,InputMatrix,OutputMatrix>
make_net_3layer_dataset(InputMatrix inputs, OutputMatrix targets) {
	return net_3layer_dataset<Net,InputMatrix,OutputMatrix>(inputs, targets);
}

template<class Net,class InputMatrix,class OutputMatrix>
struct facade {
	typedef decltype(std::declval<net_3layer_dataset<Net,InputMatrix,OutputMatrix> >().subset(std::size_t(0),std::size_t(0))) value_type;
	typedef boost::iterator_facade<net_3layer_batch_iterator<Net,InputMatrix,OutputMatrix>,
				value_type, boost::forward_traversal_tag, value_type>
		type;
};

template<class Net,class InputMatrix,class OutputMatrix>
class net_3layer_batch_iterator
	: public facade<Net,InputMatrix,OutputMatrix>::type {
public:
	typedef net_3layer_dataset<Net,InputMatrix,OutputMatrix> dataset;

	net_3layer_batch_iterator(const dataset &data, std::size_t batchsize) :
		data_(data), batchsize_(batchsize), pos_(0) {}

	// end iterator
	net_3layer_batch_iterator(const dataset &data) :
		data_(data), batchsize_(0), pos_(data.nitems()) {}

private:
	friend class boost::iterator_core_access;

	using typename facade<Net,InputMatrix,OutputMatrix>::type::value_type;

	const net_3layer_dataset<Net,InputMatrix,OutputMatrix> &data_;
	std::size_t batchsize_;
	std::size_t pos_;

	void increment() {
		pos_ += batchsize_;
		// make sure all end iterators compare equal
		if(pos_ > data_.nitems())
			pos_ = data_.nitems();
	}

	bool equal(const net_3layer_batch_iterator &other) const {
		if(pos_ == other.pos_)
			return true;

		return false;
	}

	const value_type dereference() const {
		return data_.subset(pos_, pos_ + batchsize_);
	}
};

template<class Net,class InputMatrix,class OutputMatrix>
typename net_3layer_dataset<Net,InputMatrix,OutputMatrix>::batch_iterator
net_3layer_dataset<Net,InputMatrix,OutputMatrix>::batch_begin(std::size_t batchsize) const {
	return batch_iterator(*this, batchsize);
}

template<class Net,class InputMatrix,class OutputMatrix>
typename net_3layer_dataset<Net,InputMatrix,OutputMatrix>::batch_iterator
net_3layer_dataset<Net,InputMatrix,OutputMatrix>::batch_end() const {
	return batch_iterator(*this);
}

template<class Net,class InputMatrix,class OutputMatrix>
void net_3layer_dataset<Net,InputMatrix,OutputMatrix>::shuffle() {
	Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(nitems());
	perm.setIdentity();
	std::random_device rd;
	std::mt19937 rgen(rd());
	std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), rgen);
	inputs_.matrix() = perm * inputs_.matrix();
	targets_.matrix() = perm * targets_.matrix();
}

template<class Net,class InputMatrix,class OutputMatrix>
auto net_3layer_dataset<Net,InputMatrix,OutputMatrix>::subset(std::size_t from, std::size_t to) const {
	to = std::min(to, nitems());
	return make_net_3layer_dataset<Net>(inputs_.matrix().middleRows(from, to - from), targets_.matrix().middleRows(from, to - from));
}

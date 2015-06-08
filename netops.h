#ifndef NNET_NETOPS_H
#define NNET_NETOPS_H

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/fold.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/push_back.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace netops {

template<class A> class derived_ptr;

template<class Derived>
struct expression_ptr {
	const Derived &cast() const {
		return static_cast<const Derived &>(*this);
	}

	Derived &&transfer_cast() && {
		return std::move(static_cast<Derived &&>(*this));
	}

	auto operator()() const {
		return cast()();
	}
};

template<class A>
class derived_ptr : public expression_ptr<derived_ptr<A>> {
public:
	typedef A expr_type;
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

	derived_ptr(std::unique_ptr<A> &&p) : ptr_(std::move(p)) {}
	derived_ptr(derived_ptr<A> &&p) = default;

	template<class... Args>
	auto operator()(Args&&... f) const {
		return (*ptr_)(std::forward<Args>(f)...);
	}

	template<class Input,class Weights>
	auto fprop(const Input &input, const Weights &weights) const {
		Eigen::Matrix<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> matrix;
		(*ptr_)(input, weights, [&matrix] (auto &&i, auto &&w, auto &&x) { matrix = x; });
		return matrix;
	}

	template<class Derived,class Input,class Weights,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Input &input,
			const Weights &weights, const Grads &gradients) const {
		ptr_->bprop(in, input, weights, gradients);
	}

	template<class Derived,class Input,class Weights,class Grads>
	void bprop_loss(const Eigen::MatrixBase<Derived> &in, const Input &input,
			const Weights &weights, const Grads &gradients) const {
		ptr_->bprop_loss(in, input, weights, gradients);
	}

private:
	std::unique_ptr<A> ptr_;
};

namespace detail {

/*
struct spec_hop {
	template<class State,class Index>
	auto &operator()(State &&s, Index e) const {
		return boost::fusion::at<Index>(std::forward<State>(s));
	}
};

template<class Index,class Sequence>
struct at_spec {
	typedef typename boost::fusion::result_of::fold<Index,Sequence,spec_hop>::type type;
	auto &operator()(Sequence &seq) const {
		return boost::fusion::fold(Index(), seq, spec_hop());
	}
};
*/

template<class IdxBegin,class IdxEnd>
struct at_spec {
	template<class Sequence>
	auto operator()(Sequence &&seq) const {
		typedef typename boost::mpl::deref<IdxBegin>::type idx;
		typedef typename boost::mpl::next<IdxBegin>::type next;
		//auto next_seq = boost::fusion::at<idx>(std::forward<Sequence>(seq));
		//typedef typename std::remove_reference<typename boost::fusion::result_of::at<typename std::remove_reference<Sequence>::type,idx>::type>::type next_seq;
		return at_spec<next,IdxEnd>()(boost::fusion::at<idx>(std::forward<Sequence>(seq)));
	}
};

template<class IdxEnd>
struct at_spec<IdxEnd,IdxEnd> {
	template<class Element>
	auto operator()(Element &&e) const {
		return std::forward<Element>(e);
	}
};

} // namespace detail

template<class Index,class Sequence>
auto at_spec(Sequence &&seq) {
	//return detail::at_spec<Index,Sequence>()(seq);
	typedef typename boost::mpl::begin<Index>::type begin;
	typedef typename boost::mpl::end<Index>::type end;
	return detail::at_spec<begin,end>()(std::forward<Sequence>(seq));
}

namespace expr {

namespace detail {

template<class Input,class Weights,class A,class B,class Fn>
auto binary_cont(const derived_ptr<A> &a_, const derived_ptr<B> &b_, const Input &input, const Weights &weights, const Fn &&f) {
	return a_(input, weights, [&b_, f = std::forward<const Fn>(f)] (auto &&i, auto &&w, auto &&a) {
		return b_(i, w, [a = std::forward<decltype(a)>(a), f = std::forward<const Fn>(f)] (auto &&i, auto &&w, auto &&b) {
			return std::forward<const Fn>(f)(
				std::forward<decltype(i)>(i), std::forward<decltype(w)>(w),
				std::forward<decltype(a)>(a), std::forward<decltype(b)>(b));
		});
	});
}

} // namespace detail

/*
template<class A>
class output_matrix {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

	typedef Eigen::Matrix<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> matrix_type;

	output_matrix(expression_ptr<derived_ptr<A>> &&expr) :
			expr_(std::move(expr).transfer_cast()){}

	template<class Data>
	const matrix_type &operator()(const Data &data) {
		expr_(data, [this] (auto &&data, auto &&x) { this->matrix_ = x; });
		return matrix_;
	}

	template<class Derived,class Data,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Data &data, const Grads &grads) const {
		expr_.bprop(in, data, grads);
	}

private:
	derived_ptr<A> expr_;
	matrix_type matrix_;
};
*/

template<class Index,class Spec>
class input_matrix {
private:
	typedef decltype(at_spec<Index>(std::declval<Spec>())) spec_type;
	//typedef typename ::netops::detail::at_spec<Index,Spec>::type spec_type;

public:
	typedef typename spec_type::float_type F;
	enum {
		RowsAtCompileTime = spec_type::RowsAtCompileTime,
		ColsAtCompileTime = spec_type::ColsAtCompileTime,
		StorageOrder = spec_type::StorageOrder
	};

	input_matrix() {}

	template<class Input,class Weights>
	const auto &operator()(const Input &input, const Weights &weights) {
		return at_spec<Index>(input);
	}

	template<class Input,class Weights,class Fn>
	auto operator()(const Input &input, const Weights &weights, Fn &&f) {
		return std::forward<Fn>(f)(input, weights, at_spec<Index>(input));
	}

	template<class Derived,class Input,class Weights,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Input &input, const Weights &weights, const Grads &gradients) const {}
};

template<class Index,class Spec>
class weight_matrix {
private:
	typedef decltype(at_spec<Index>(std::declval<Spec>())) spec_type;
	//typedef typename ::netops::detail::at_spec<Index,Spec>::type spec_type;

public:
	typedef typename spec_type::float_type F;
	enum {
		RowsAtCompileTime = spec_type::RowsAtCompileTime,
		ColsAtCompileTime = spec_type::ColsAtCompileTime,
		StorageOrder = spec_type::StorageOrder
	};

	weight_matrix() {}

	template<class Input,class Weights>
	const auto &operator()(const Input &input, const Weights &weights) {
		return at_spec<Index>(weights);
	}

	template<class Input,class Weights,class Fn>
	auto operator()(const Input &input, const Weights &weights, Fn &&f) {
		return std::forward<Fn>(f)(input, weights, at_spec<Index>(weights));
	}

	template<class Derived,class Input,class Weights,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Input &input, const Weights &weights, Grads &gradients) const {
		at_spec<Index>(gradients) += in;
	}
};

template<class... Args>
class concat {
private:
	typedef typename boost::fusion::vector<derived_ptr<Args>...> expr_vector;
	typedef typename boost::fusion::result_of::value_at_c<expr_vector,0>::type first_arg_type;

public:
	typedef typename first_arg_type::F F;
	enum {
		RowsAtCompileTime = first_arg_type::RowsAtCompileTime,
		ColsAtCompileTime = Eigen::Dynamic,
		StorageOrder = first_arg_type::StorageOrder
	};

private:
	typedef Eigen::Matrix<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> matrix_type;

public:
	concat(expression_ptr<derived_ptr<Args>> &&... args) :
		exprs_(std::move(args).transfer_cast()...) {}

	template<class Input,class Weights,class Fn>
	auto operator()(const Input &input, const Weights &weights, Fn &&f) {
		namespace fusion = boost::fusion;
		auto colit = cols_.begin();
		std::size_t nrows;
		std::size_t ncols = fusion::fold(exprs_, std::size_t(0), [&input, &weights, &nrows, &colit] (std::size_t s, auto &e) {
			std::size_t c;
			e(input, weights, [&c, &nrows] (auto &&i, auto &&w, auto &&a) { nrows = a.rows(); c = a.cols(); });
			*(colit++) = c;
			return s + c;
		});
		concat_.resize(nrows, ncols);
		fusion::fold(exprs_, std::size_t(0), [this, &input, &weights] (std::size_t s, auto &e) {
			std::size_t c;
			e(input, weights, [this, s, &c] (auto &&i, auto &&w, auto &&a) {
				this->concat_.middleCols(s, a.cols()) = std::forward<decltype(a)>(a);
				c = a.cols();
			});
			return s + c;
		});
		return std::forward<Fn>(f)(input, weights, concat_);
	}

	template<class Derived,class Input,class Weights,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Input &input, const Weights &weights, const Grads &gradients) const {
		const auto &eval_in = in.eval();
		auto colit = cols_.begin();
		boost::fusion::fold(exprs_, std::size_t(0), [&eval_in, &input, &weights, &gradients, &colit] (std::size_t s, auto &e) {
			e.bprop(eval_in.middleCols(s, *colit), input, weights, gradients);
			return s + *(colit++);
		});
	}

private:
	expr_vector exprs_;
	std::array<std::size_t,sizeof...(Args)> cols_;
	matrix_type concat_;
};

template<class A,class B>
class rowwise_add {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

	rowwise_add(expression_ptr<derived_ptr<A>> &&a, expression_ptr<derived_ptr<B>> &&b) :
		a_(std::move(a).transfer_cast()), b_(std::move(b).transfer_cast()) {}

	template<class Input,class Weights,class Fn>
	auto operator()(const Input &input, const Weights &weights, Fn &&f) {
		return detail::binary_cont(a_, b_, input, weights,
			[f = std::forward<Fn>(f)] (auto &&i, auto &&w, auto &&a, auto &&b) {
				return f(std::forward<decltype(i)>(i), std::forward<decltype(w)>(w),
					std::forward<decltype(a)>(a).rowwise() + std::forward<decltype(b)>(b));
		});
	}

	template<class Derived,class Input,class Weights,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Input &input, const Weights &weights, const Grads &gradients) const {
		const auto &eval_in = in.eval();
		a_.bprop(eval_in, input, weights, gradients);
		b_.bprop(eval_in.colwise().sum(), input, weights, gradients);
	}

private:
	derived_ptr<A> a_;
	derived_ptr<B> b_;
};

template<class A,class B>
class matmul {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = B::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

	matmul(expression_ptr<derived_ptr<A>> &&a, expression_ptr<derived_ptr<B>> &&b) :
			a_(std::move(a).transfer_cast()), b_(std::move(b).transfer_cast()) {
	}

	template<class Input,class Weights,class Fn>
	auto operator()(const Input &input, const Weights &weights, Fn &&f) {
		detail::binary_cont(a_, b_, input, weights, [f = std::forward<Fn>(f)] (auto &&i, auto &&w, auto &&a, auto &&b) {
			return std::forward<decltype(f)>(f)(
				std::forward<decltype(i)>(i), std::forward<decltype(w)>(w),
				std::forward<decltype(a)>(a) * std::forward<decltype(b)>(b));
		});
	}

	template<class Derived,class Input,class Weights,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Input &input, const Weights &weights, const Grads &gradients) const {
		const auto &eval_in = in.eval();
		b_(input, weights, [this, &eval_in, &gradients] (auto &&i, auto &&w, auto &&b) {
			this->a_.bprop(eval_in * b.transpose(), i, w, gradients);
		});
		a_(input, weights, [this, &eval_in, &gradients] (auto &&i, auto &&w, auto &&a) {
			this->b_.bprop(a.transpose() * eval_in, i, w, gradients);
		});
	}

private:
	derived_ptr<A> a_;
	derived_ptr<B> b_;
};

template<class A>
class logistic_sigmoid {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

	logistic_sigmoid(expression_ptr<derived_ptr<A>> &&a) :
			a_(std::move(a).transfer_cast()) {}

	template<class Input,class Weights,class Fn>
	auto operator()(const Input &input, const Weights &weights, Fn &&f) {
		a_(input, weights, [this] (auto &&i, auto &&w, auto &&a) { this->result_ = F(1) / (F(1) + (-a).array().exp()); });
		return std::forward<Fn>(f)(input, weights, result_.matrix());
	}

	template<class Derived,class Input,class Weights,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Input &input, const Weights &weights, const Grads &gradients) const {
		a_.bprop((in.array() * result_ * (1 - result_)).matrix(), input, weights, gradients);
	}

private:
	derived_ptr<A> a_;
	typedef Eigen::Array<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> array_type;
	array_type result_;
};

template<class A>
class softmax_crossentropy {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

private:
	typedef Eigen::Array<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> array_type;

public:
	softmax_crossentropy(expression_ptr<derived_ptr<A>> &&a) :
			a_(std::move(a).transfer_cast()) {}

	template<class Input,class Weights,class Fn>
	auto operator()(const Input &input, const Weights &weights, Fn &&f) {
		array_type eval_a;
		a_(input, weights, [&eval_a] (auto &&i, auto &&w, auto &&a) { eval_a = a.array(); });
		array_type t = (eval_a.colwise() - eval_a.rowwise().maxCoeff()).exp();
		result_ = t.colwise() / t.rowwise().sum();
		return std::forward<Fn>(f)(input, weights, result_.matrix());
	}

	template<class Derived,class Input,class Weights,class Grads>
	void bprop_loss(const Eigen::MatrixBase<Derived> &targets, const Input &input, const Weights &weights, const Grads &gradients) const {
		a_.bprop(result_.matrix() - targets, input, weights, gradients);
	}

private:
	derived_ptr<A> a_;
	array_type result_;
};

/*
template<class MapIdx,class A>
class nn6_softmax {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

private:
	typedef typename Eigen::SparseMatrix<F,Eigen::RowMajor> map_matrix; // RowMajor is important here

public:
	nn6_softmax(expression_ptr<derived_ptr<A>> &&a) :
			a_(std::move(a).transfer_cast()) {}

	template<class Data,class Fn>
	auto operator()(const Data &data, Fn &&f) {
		const auto &weights = detail::at_spec<WeightIdx,Data>()(data);
		const auto &mapping = detail::at_spec<MapIdx,Data>()(data); // a vector with the number of rows for each example

		Eigen::Matrix<F,Eigen::Dynamic,1,StorageOrder> maxcoeff;
		a_(data, [this, &weights, &mapping, &maxcoeff] (auto &&a) {
			const auto &smin = (a * weights).eval();
			this->mapmat_.resize(mapping.rows(), a.rows());
			this->mapmat_.reserve(mapping);
			maxcoeff.resize(mapping.rows());
			maxcoeff.setConstant(-std::numeric_limits<F>::infinity());
			for(int row = 0, col = 0, n = 0; col < a.rows(); ++col, ++n) {
				if(n > mapping(row))
					n = 0, ++row;
				if(a(col) > maxcoeff(row))
					maxcoeff(row) = a(col);
				this->mapmat_.insert(row, col, a(col));
			}
		});

		Eigen::Matrix<F,Eigen::Dynamic,1,StorageOrder> sum(mapmat_.rows());
		sum.setZero();
		for(int i = 0; i < mapmat_.outerSize(); ++i)
			for(typename map_matrix::InnerIterator it(mapmat_, i); it; ++it) {
				using std::exp;
				F val = exp(it.value() - maxcoeff(it.row()));
				sum(it.row()) += val;
				it.value() = val;
			}

		for(int i = 0; i < mapmat_.outerSize(); ++i)
			for(typename map_matrix::InnerIterator it(mapmat_, i); it; ++it)
				it.value() /= sum(it.row());

		return std::forward<Fn>(f)(mapmat_);
	}

	template<class Derived,class Data,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Data &data, const Grads &gradients) const {
		const auto &eval_in = in.rowwise().sum().eval();

		Grads subgrads(gradients);
		subgrads.array().setZero();
		a_.bprop(Eigen::Matrix<F,Eigen::Dynamic,1,StorageOrder>::Ones(eval_in.rows()), data, subgrads);
		
		// fix softmax derivative
	}

private:
	derived_ptr<A> a_;
	map_matrix mapmat_;
};
*/

template<class MapIdx,class A,class B>
class nn6_combiner {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

private:
	typedef typename Eigen::SparseMatrix<F,Eigen::RowMajor> map_matrix; // RowMajor is important here

public:
	nn6_combiner(expression_ptr<derived_ptr<A>> &&ant, expression_ptr<derived_ptr<B>> &&map) :
			ant_(std::move(ant).transfer_cast()), map_(std::move(map).transfer_cast()) {}

	template<class Input,class Weights,class Fn>
	auto operator()(const Input &input, const Weights &weights, Fn &&f) {
		Eigen::Matrix<F,Eigen::Dynamic,1> map;
		map_(input, weights, [&map] (auto &&i, auto &&w, auto &&m) { map = m; });
		const auto &antmap = at_spec<MapIdx>(input);
		mapmat_.resize(antmap.rows(), map.rows());
		mapmat_.reserve(antmap);
		for(int i = 0, c = 0; i < antmap.rows(); i++)
			for(int j = 0; j < antmap(i); j++, c++)
				mapmat_.insert(i,c) = map(c);
		return ant_(input, weights, [this, f = std::forward<Fn>(f)] (auto &&i, auto &&w, auto &&ant) {
			//return std::forward<decltype(f)>(f)(std::forward<decltype(d)>(d),
				//this->mapmat_ * std::forward<decltype(ant)>(ant));
			this->antin_ = ant;
			return std::forward<decltype(f)>(f)(
				std::forward<decltype(i)>(i), std::forward<decltype(w)>(w), this->mapmat_ * this->antin_);
		});
	}

	template<class Derived,class Input,class Weights,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Input &input, const Weights &weights, const Grads &gradients) const {
		const auto &eval_in = in.eval();
		ant_.bprop(mapmat_.transpose() * eval_in, input, weights, gradients);
		map_matrix backmat(mapmat_);
		for(int i = 0; i < backmat.outerSize(); ++i)
			for(typename map_matrix::InnerIterator it(backmat, i); it; ++it)
				it.valueRef() = 1;
		map_.bprop((backmat.transpose() * eval_in).cwiseProduct(antin_).rowwise().sum(), input, weights, gradients);
	}

private:
	derived_ptr<A> ant_;
	derived_ptr<B> map_;
	Eigen::Matrix<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> antin_;
	map_matrix mapmat_;
};

template<class MapIdx,class A>
class max_pooling {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = Eigen::Dynamic,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

private:
	typedef typename Eigen::Matrix<F,Eigen::Dynamic,ColsAtCompileTime,StorageOrder> mask_matrix;

public:
	max_pooling(expression_ptr<derived_ptr<A>> &&a) : a_(std::move(a).transfer_cast()) {}

	template<class Input,class Weights,class Fn>
	auto operator()(const Input &input, const Weights &weights, Fn &&f) {
		const auto &map = at_spec<MapIdx>(input);
		return a_(input, weights, [this, &map, f = std::forward<Fn>(f)] (auto &&i, auto &&w, auto &&a) {
			this->maxmask_.resize(a.rows(), a.cols());
			this->maxmask_.setZero();
			Eigen::Matrix<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> out;
			out.resize(a.rows(), a.cols());
			out.setConstant(-std::numeric_limits<F>::infinity());
			for(int i = 0, k = 0; i < a.rows(); i++) {
				Eigen::Matrix<int,1,ColsAtCompileTime,StorageOrder> idx;
				idx.resize(a.cols());
				idx.setZero();
				for(int j = 0; j < map(i); j++, k++) {
					for(int l = 0; l < a.cols(); l++) {
						if(a(k,l) > out(i,l)) {
							out(i,l) = a(k,l);
							idx(l) = k;
						}
					}
				}
				for(int j = 0; j < idx.cols(); j++)
					this->maxmask_(i,idx(j)) = 1;
			}
			return std::forward<Fn>(f)(std::forward<decltype(i)>(i), std::forward<decltype(w)>(w), out);
		});
	}

	template<class Derived,class Input,class Weights,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Input &input, const Weights &weights, const Grads &gradients) const {
		const auto &eval_in = in.eval();
		Eigen::Matrix<F,A::RowsAtCompileTime,ColsAtCompileTime,StorageOrder> outgrads;
		outgrads.resizeLike(maxmask_);
		const auto &map = at_spec<MapIdx>(input);
		for(int i = 0, k = 0; i < eval_in.rows(); i++)
			for(int j = 0; j < map(i); j++, k++)
				outgrads.row(k) = eval_in.row(i);
		a_.bprop(maxmask_.cwiseProduct(outgrads), input, weights, gradients);
	}

private:
	derived_ptr<A> a_;
	mask_matrix maxmask_;
};

} // namespace expr

/*
template<class A>
derived_ptr<expr::output_matrix<A>>
eval(expression_ptr<derived_ptr<A>> &&a) {
	return std::make_unique<expr::output_matrix<A>>(std::move(a).transfer_cast());
}
*/

template<class Index,class Spec>
derived_ptr<expr::input_matrix<Index,Spec>>
input_matrix(const Spec &spec) {
	return std::make_unique<expr::input_matrix<Index,Spec>>();
}

template<class Index,class Spec>
derived_ptr<expr::weight_matrix<Index,Spec>>
weight_matrix(const Spec &spec) {
	return std::make_unique<expr::weight_matrix<Index,Spec>>();
}

template<class... Args>
derived_ptr<expr::concat<Args...>>
concat(expression_ptr<derived_ptr<Args>> &&... args) {
	return std::make_unique<expr::concat<Args...>>(std::move(args).transfer_cast()...);
}

template<class A,class B>
std::enable_if_t<B::RowsAtCompileTime==1,derived_ptr<expr::rowwise_add<A,B>>>
operator+(expression_ptr<derived_ptr<A>> &&a, expression_ptr<derived_ptr<B>> &&b) {
	return std::make_unique<expr::rowwise_add<A,B>>(std::move(a).transfer_cast(), std::move(b).transfer_cast());
}

template<class A,class B>
derived_ptr<expr::matmul<A,B>>
operator*(expression_ptr<derived_ptr<A>> &&a, expression_ptr<derived_ptr<B>> &&b) {
	return std::make_unique<expr::matmul<A,B>>(std::move(a).transfer_cast(), std::move(b).transfer_cast());
}

template<class A>
derived_ptr<expr::logistic_sigmoid<A>>
logistic_sigmoid(expression_ptr<derived_ptr<A>> &&a) {
	return std::make_unique<expr::logistic_sigmoid<A>>(std::move(a).transfer_cast());
}

template<class A>
derived_ptr<expr::softmax_crossentropy<A>>
softmax_crossentropy(expression_ptr<derived_ptr<A>> &&a) {
	return std::make_unique<expr::softmax_crossentropy<A>>(std::move(a).transfer_cast());
}

template<class MapIdx,class Ant,class Map>
derived_ptr<expr::nn6_combiner<MapIdx,Ant,Map>>
nn6_combiner(expression_ptr<derived_ptr<Ant>> &&ant, expression_ptr<derived_ptr<Map>> &&map) {
	return std::make_unique<expr::nn6_combiner<MapIdx,Ant,Map>>(std::move(ant).transfer_cast(), std::move(map).transfer_cast());
}

template<class MapIdx,class A>
derived_ptr<expr::max_pooling<MapIdx,A>>
max_pooling(expression_ptr<derived_ptr<A>> &&a) {
	return std::make_unique<expr::max_pooling<MapIdx,A>>(std::move(a).transfer_cast());
}

/*
template<class WIdx,class BIdx,class A,class Spec>
auto
linear_layer(const Spec &spec, expression_ptr<derived_ptr<A>> &&a) {
	return std::move(a).transfer_cast() * weight_matrix<WIdx>(spec) + weight_matrix<BIdx>(spec);
}
*/

template<class Idx,class A,class Spec>
auto
linear_layer(const Spec &spec, expression_ptr<derived_ptr<A>> &&a) {
	typedef typename boost::mpl::push_back<Idx,boost::mpl::int_<0>>::type WIdx;
	typedef typename boost::mpl::push_back<Idx,boost::mpl::int_<1>>::type BIdx;
	return std::move(a).transfer_cast() * weight_matrix<WIdx>(spec) + weight_matrix<BIdx>(spec);
}

} // namespace netops

#endif

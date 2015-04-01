#ifndef NNET_NNET_H
#define NNET_NNET_H

#include <cmath>
#include <type_traits>
#include <utility>

#include <Eigen/Core>

namespace nnet {

// a printable representation of the value
// (will be specialised for AD types)
template<class C>
struct value_functor {
	typedef const C &result_type;
	result_type operator()(const C &in) const {
		return in;
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

} // namespace nnet

#endif

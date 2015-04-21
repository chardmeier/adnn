#ifndef NNET_NETOPS_H
#define NNET_NETOPS_H

#include <memory>
#include <type_traits>

#include <boost/fusion/include/at.hpp>
#include <boost/fusion/include/fold.hpp>
#include <boost/fusion/include/mpl.hpp>

#include <Eigen/Core>

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

	template<class Data>
	auto fprop(const Data &data) const {
		Eigen::Matrix<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> matrix;
		(*ptr_)(data, [&matrix] (auto &&data, auto &&x) { matrix = x; });
		return matrix;
	}

	template<class Derived,class Data,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Data &data, const Grads &gradients) const {
		ptr_->bprop(in, data, gradients);
	}

	template<class Derived,class Data,class Grads>
	void bprop_loss(const Eigen::MatrixBase<Derived> &in, const Data &data, const Grads &gradients) const {
		ptr_->bprop_loss(in, data, gradients);
	}

private:
	std::unique_ptr<A> ptr_;
};

namespace expr {

namespace detail {

template<class Data,class A,class B,class Fn>
auto binary_cont(const derived_ptr<A> &a_, const derived_ptr<B> &b_, const Data &data, const Fn &&f) {
	return a_(data, [&b_, f = std::forward<const Fn>(f)] (auto &&data, auto &&a) {
		return b_(data, [a = std::forward<decltype(a)>(a), f = std::forward<const Fn>(f)] (auto &&data, auto &&b) {
			return std::forward<const Fn>(f)(std::forward<decltype(data)>(data),
				std::forward<decltype(a)>(a), std::forward<decltype(b)>(b));
		});
	});
}

struct spec_hop {
	template<class State,class Index>
	auto operator()(State &&s, Index e) const {
		return boost::fusion::at<Index>(std::forward<State>(s));
	}
};

template<class Index,class Sequence>
struct at_spec {
	typedef typename boost::fusion::result_of::fold<Index,Sequence,spec_hop>::type type;
	auto operator()(Sequence &seq) const {
		return boost::fusion::fold(Index(), seq, spec_hop());
	}
};

} // namespace detail

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

template<class Index,class Spec>
class input_matrix {
private:
	typedef typename detail::at_spec<Index,Spec>::type spec_type;

public:
	typedef typename spec_type::float_type F;
	enum {
		RowsAtCompileTime = spec_type::RowsAtCompileTime,
		ColsAtCompileTime = spec_type::ColsAtCompileTime,
		StorageOrder = spec_type::StorageOrder
	};

	input_matrix() {}

	template<class Data>
	const auto &operator()(const Data &data) {
		return detail::at_spec<Index,const Data>(data);
	}

	template<class Data,class Fn>
	auto operator()(const Data &data, Fn &&f) {
		return std::forward<Fn>(f)(data, detail::at_spec<Index,const Data>()(data));
	}

	template<class Derived,class Data,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Data &data, const Grads &gradients) const {}
};

template<class Index,class Spec>
class weight_matrix {
private:
	typedef typename detail::at_spec<Index,Spec>::type spec_type;

public:
	typedef typename spec_type::float_type F;
	enum {
		RowsAtCompileTime = spec_type::RowsAtCompileTime,
		ColsAtCompileTime = spec_type::ColsAtCompileTime,
		StorageOrder = spec_type::StorageOrder
	};

	weight_matrix() {}

	template<class Data>
	const auto &operator()(const Data &data) {
		return detail::at_spec<Index,const Data>(data);
	}

	template<class Data,class Fn>
	auto operator()(const Data &data, Fn &&f) {
		return std::forward<Fn>(f)(data, detail::at_spec<Index,const Data>()(data));
	}

	template<class Derived,class Data,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Data &data, const Grads &gradients) const {
		detail::at_spec<Index,const Grads>()(gradients) += in;
	}
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

	template<class Data,class Fn>
	auto operator()(const Data &data, Fn &&f) {
		return detail::binary_cont(a_, b_, data,
			[f = std::forward<Fn>(f)] (auto &&data, auto &&a, auto &&b) {
				return f(std::forward<decltype(data)>(data),
					std::forward<decltype(a)>(a).rowwise() + std::forward<decltype(b)>(b));
		});
	}

	template<class Derived,class Data,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Data &data, const Grads &gradients) const {
		const auto &eval_in = in.eval();
		a_.bprop(eval_in, data, gradients);
		b_.bprop(eval_in.colwise().sum(), data, gradients);
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

	template<class Data,class Fn>
	auto operator()(const Data &data, Fn &&f) {
		detail::binary_cont(a_, b_, data, [f = std::forward<Fn>(f)] (auto &&data, auto &&a, auto &&b) {
			return std::forward<decltype(f)>(f)(std::forward<decltype(data)>(data),
				std::forward<decltype(a)>(a) * std::forward<decltype(b)>(b));
		});
	}

	template<class Derived,class Data,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Data &data, const Grads &gradients) const {
		const auto &eval_in = in.eval();
		b_(data, [this, &eval_in, &gradients] (auto &&data, auto &&b) { this->a_.bprop(eval_in * b.transpose(), data, gradients); });
		a_(data, [this, &eval_in, &gradients] (auto &&data, auto &&a) { this->b_.bprop(a.transpose() * eval_in, data, gradients); });
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

	template<class Data,class Fn>
	auto operator()(const Data &data, Fn &&f) {
		a_(data, [this] (auto &&data, auto &&a) { this->result_ = F(1) / (F(1) + (-a).array().exp()); });
		return std::forward<Fn>(f)(data, result_.matrix());
	}

	template<class Derived,class Data,class Grads>
	void bprop(const Eigen::MatrixBase<Derived> &in, const Data &data, const Grads &gradients) const {
		a_.bprop((in.array() * result_ * (1 - result_)).matrix(), data, gradients);
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

	template<class Data,class Fn>
	auto operator()(const Data &data, Fn &&f) {
		array_type eval_a;
		a_(data, [&eval_a] (auto &&data, auto &&a) { eval_a = a.array(); });
		array_type t = (eval_a.colwise() - eval_a.rowwise().maxCoeff()).exp();
		result_ = t.colwise() / t.rowwise().sum();
		return std::forward<Fn>(f)(data, result_.matrix());
	}

	template<class Derived,class Data,class Grads>
	void bprop_loss(const Eigen::MatrixBase<Derived> &targets, const Data &data, const Grads &gradients) const {
		a_.bprop(result_.matrix() - targets, data, gradients);
	}

private:
	derived_ptr<A> a_;
	array_type result_;
};

} // namespace expr

template<class A>
derived_ptr<expr::output_matrix<A>>
eval(expression_ptr<derived_ptr<A>> &&a) {
	return std::make_unique<expr::output_matrix<A>>(std::move(a).transfer_cast());
}

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

template<class WIdx,class BIdx,class A,class Spec>
auto
linear_layer(const Spec &spec, expression_ptr<derived_ptr<A>> &&a) {
	return std::move(a).transfer_cast() * weight_matrix<WIdx>(spec) + weight_matrix<BIdx>(spec);
}

} // namespace netops

#endif

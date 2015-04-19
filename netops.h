#ifndef NNET_NETOPS_H
#define NNET_NETOPS_H

#include <memory>
#include <type_traits>

#include <boost/fusion/include/at.hpp>
#include <boost/mpl/front.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/vector.hpp>

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

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {
		ptr_->bprop(in);
	}

	template<class Derived>
	void bprop_loss(const Eigen::MatrixBase<Derived> &in) const {
		ptr_->bprop_loss(in);
	}

private:
	std::unique_ptr<A> ptr_;
};

namespace expr {

namespace detail {

template<class Data,class A,class B,class F>
auto binary_cont(const derived_ptr<A> &a_, const derived_ptr<B> &b_, const Data &data, const F &&f) {
	return a_(data, [&b_, &data, f = std::forward<const F>(f)] (auto &&a) {
		return b_(data, [a = std::forward<decltype(a)>(a), f = std::forward<const F>(f)] (auto &&b) {
			return std::forward<const F>(f)(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b));
		});
	});
}

template<class Index,class Sequence>
struct at_spec {
	using namespace fusion = boost::fusion;
	using namespace mpl = boost::mpl;

	typedef typename mpl::front<Index> head;
	typedef typename mpl::pop_front<Index> tail;

	typedef typename fusion::result_of::at<head,Sequence>::type next_type;
	typedef typename at_spec<tail,next_type>::result_type result_type;

	result_type operator()(Sequence &seq) const {
		return at_spec<tail,next_type>()(fusion::at<head>(seq));
	}
};

template<class Data>
struct at_spec<boost::mpl::vector<>,Data> {
	typedef Data result_type;

	result_type operator()(Data &data) const {
		return data;
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
	const matrix_type &operator()(const Data &data) const {
		expr_(data, [this] (auto &&x) { this->matrix_ = x; });
		return matrix_;
	}

	template<class Derived,class Data>
	void bprop(const Eigen::MatrixBase<Derived> &in, Data &data) const {
		expr_.bprop(in, data);
	}

private:
	derived_ptr<A> expr_;
	matrix_type matrix_;
};

template<class Index,class Spec>
class input_matrix {
private:
	typedef typename at_spec<Index,Spec>::type spec_type;

public:
	typedef typename spec_type::float_type F;
	enum {
		RowsAtCompileTime = spec_type::RowsAtCompileTime,
		ColsAtCompileTime = spec_type::ColsAtCompileTime,
		StorageOrder = spec_type::StorageOrder
	};

	input_matrix() {}

	template<class Data>
	const auto &operator()(const Data &data) const {
		return at_spec<Index>(data);
	}

	template<class Data,class F>
	auto operator()(const Data &data, F &&f) const {
		return std::forward<F>(f)(at_spec<Index>(data));
	}

	template<class Derived,class Data>
	void bprop(const Eigen::MatrixBase<Derived> &in, Data &gradients) const {}
};

template<class Index,class Spec>
class weight_matrix {
private:
	typedef typename at_spec<Index,Spec>::type spec_type;

public:
	typedef typename spec_type::float_type F;
	enum {
		RowsAtCompileTime = spec_type::RowsAtCompileTime,
		ColsAtCompileTime = spec_type::ColsAtCompileTime,
		StorageOrder = spec_type::StorageOrder
	};

	weight_matrix() {}

	template<class Data>
	const auto &operator()(const Data &data) const {
		return at_spec<Index>(data);
	}

	template<class Data,class F>
	auto operator()(const Data &data, F &&f) const {
		return std::forward<F>(f)(at_spec<Index>(data));
	}

	template<class Derived,class Data>
	void bprop(const Eigen::MatrixBase<Derived> &in, Data &gradients) const {
		at_spec<Index>(gradients) += in;
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

	template<class F>
	auto operator()(const Data &data, F &&f) const {
		return detail::binary_cont(a_, b_, data,
			[f = std::forward<F>(f)] (auto &&a, auto &&b) {
				return f(std::forward<decltype(a)>(a).rowwise() +
					std::forward<decltype(b)>(b));
		});
	}

	template<class Derived,class Data>
	void bprop(const Eigen::MatrixBase<Derived> &in, Data &gradients) const {
		const auto &eval_in = in.eval();
		a_.bprop(eval_in, gradients);
		b_.bprop(eval_in.colwise().sum(), gradients);
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

	template<class Data,class F>
	auto operator()(const Data &data, F &&f) const {
		detail::binary_cont(a_, b_, data, [f = std::forward<F>(f)] (auto &&a, auto &&b) {
			return std::forward<F>(f)(std::forward<decltype(a)>(a) * std::forward<decltype(b)>(b));
		});
	}

	template<class Derived,class Data>
	void bprop(const Eigen::MatrixBase<Derived> &in, Data &gradients) const {
		const auto &eval_in = in.eval();
		b_([this, &eval_in, &gradients] (auto &&b) { this->a_.bprop(eval_in * b.transpose(), gradients); });
		a_([this, &eval_in, &gradients] (auto &&a) { this->b_.bprop(a.transpose() * eval_in, gradients); });
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

	template<class Data,class F>
	auto operator()(const Data &data, F &&f) const {
		a_(data, [this] (auto &&a) { this->result_ = F(1) / (F(1) + (-a).array().exp()); });
		return std::forward<F>(f)(result_.matrix());
	}

	template<class Derived,class Data>
	void bprop(const Eigen::MatrixBase<Derived> &in, Data &gradients) const {
		a_.bprop((in.array() * result_ * (1 - result_)).matrix(), gradients);
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

	template<class Data,class F>
	auto operator()(const Data &data, F &&f) const {
		array_type eval_a;
		a_(data, [&eval_a] (auto &&a) { eval_a = a.array(); });
		array_type t = (eval_a.colwise() - eval_a.rowwise().maxCoeff()).exp();
		result_ = t.colwise() / t.rowwise().sum();
		return std::forward<F>(f)(result_.matrix());
	}

	template<class Derived,class Data>
	void bprop_loss(const Eigen::MatrixBase<Derived> &targets, Data &gradients) const {
		a_.bprop(result_.matrix() - targets, gradients);
	}

private:
	derived_ptr<A> a_;
	array_type result_;
};

} // namespace expr

template<class A>
derived_ptr<expr::output_matrix<A>>
eval(expression_ptr<derived_ptr<A>> &&a) {
	std::make_unique<expr::output_matrix<A>>(std::move(a).transfer_cast());
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

} // namespace netops

#endif

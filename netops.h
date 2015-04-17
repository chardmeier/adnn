#ifndef NNET_NETOPS_H
#define NNET_NETOPS_H

#include <memory>
#include <type_traits>

#include <Eigen/Core>

namespace netops {

namespace expr {

namespace detail {

/*
template<class Derived,class Base,class Del>
std::unique_ptr<Derived,Del>
cast(std::unique_ptr<Base,Del> &&p) {
	auto d = static_cast<Derived *>(p.release());
	return std::unique_ptr<Derived,Del>(d, std::move(p.get_deleter()));
}
*/

} // namespace detail

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

private:
	std::unique_ptr<A> ptr_;
};

namespace detail {

template<class A,class B,class F>
auto binary_cont(const derived_ptr<A> &a_, const derived_ptr<B> &b_, const F &&f) {
	return a_([&b_, f = std::forward<const F>(f)] (auto &&a) {
		return b_([a = std::forward<decltype(a)>(a), f = std::forward<const F>(f)] (auto &&b) {
			return std::forward<const F>(f)(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b));
		});
	});
}

template<class A>
auto cast_if_necessary(A &&a) {
	return std::forward<A>(a);
}

template<class D>
D &&cast_if_necessary(expression_ptr<D> &&p) {
	return std::move(p).transfer_cast();
}

template<class A,class... Args>
derived_ptr<A> make_derived(Args&&... args) {
	return derived_ptr<A>(std::make_unique<A>(cast_if_necessary(std::forward<Args>(args)) ...));
}

} // namespace detail

template<class A> class output_matrix;

template<class Derived>
struct expression {};

template<class A>
class output_matrix : public expression<output_matrix<A>> {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

	typedef Eigen::Matrix<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> matrix_type;

	output_matrix(expression_ptr<derived_ptr<A>> &&expr) :
			expr_(std::move(expr).transfer_cast()){
		expr_([this] (auto &&x) { this->matrix_ = x; });
		std::cerr << "Created output_matrix.\n";
	}

	const matrix_type &operator()() const {
		return matrix_;
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {
		expr_.bprop(in);
	}

	~output_matrix() {
		std::cerr << "Destroyed output_matrix.\n";
	}

private:
	derived_ptr<A> expr_;
	matrix_type matrix_;
};

/*
template<class Derived>
inline output_matrix<Derived> expression<Derived>::eval() const {
	return output_matrix<Derived>(cast());
}
*/

template<class A>
class input_matrix : public expression<input_matrix<A>> {
public:
	typedef typename A::Scalar F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor
	};

	input_matrix(const A &mat) : mat_(mat) {
		std::cerr << "Created input_matrix.\n";
	}

	template<class F>
	auto operator()(F &&f) const {
		return std::forward<F>(f)(mat_);
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {}

	~input_matrix() {
		std::cerr << "Destroyed input_matrix.\n";
	}

private:
	const A &mat_;
};

template<class A,class B>
class weight_matrix : public expression<weight_matrix<A,B>> {
public:
	typedef typename A::Scalar F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor
	};

	weight_matrix(const Eigen::MatrixBase<A> &weights, const Eigen::MatrixBase<B> &gradients) :
			weights_(weights), gradients_(const_cast<B&>(gradients.derived())) {
		gradients_.setZero();
		std::cerr << "Created weight_matrix.\n";
	}

	template<class F>
	auto operator()(F &&f) const {
		return std::forward<F>(f)(weights_);
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {
		gradients_ += in;
	}

	~weight_matrix() {
		std::cerr << "Destroyed weight_matrix.\n";
	}

private:
	const Eigen::MatrixBase<A> &weights_;
	B &gradients_;
};

template<class A,class B>
class rowwise_add : public expression<rowwise_add<A,B>> {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

	rowwise_add(expression_ptr<derived_ptr<A>> &&a, expression_ptr<derived_ptr<B>> &&b) :
		a_(std::move(a).transfer_cast()), b_(std::move(b).transfer_cast()) {
		std::cerr << "Created rowwise_add.\n";
	}

	template<class F>
	auto operator()(F &&f) const {
		return detail::binary_cont(a_, b_,
			[f = std::forward<F>(f)] (auto &&a, auto &&b) {
				return f(std::forward<decltype(a)>(a).rowwise() +
					std::forward<decltype(b)>(b));
		});
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {
		const auto &eval_in = in.eval();
		a_.bprop(eval_in);
		b_.bprop(eval_in.colwise().sum());
	}

	~rowwise_add() {
		std::cerr << "Destroyed rowwise_add.\n";
	}

private:
	derived_ptr<A> a_;
	derived_ptr<B> b_;
};

template<class A,class B>
class matmul : public expression<matmul<A,B>> {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = B::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

	matmul(expression_ptr<derived_ptr<A>> &&a, expression_ptr<derived_ptr<B>> &&b) :
			a_(std::move(a).transfer_cast()), b_(std::move(b).transfer_cast()) {
		detail::binary_cont(a_, b_, [this] (auto &&a, auto &&b) {
			this->result_ = std::forward<decltype(a)>(a) * std::forward<decltype(b)>(b);
		});
		std::cerr << "Created matmul.\n";
	}

	template<class F>
	auto operator()(F &&f) const {
		return std::forward<F>(f)(result_);
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {
		const auto &eval_in = in.eval();
		a_.bprop(eval_in * b_().transpose());
		b_.bprop(a_().transpose() * eval_in);
	}

	~matmul() {
		std::cerr << "Destroyed matmul.\n";
	}

private:
	derived_ptr<A> a_;
	derived_ptr<B> b_;
	Eigen::Matrix<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> result_;
};

/*
template<class A>
class logistic_sigmoid : public expression<logistic_sigmoid<A>> {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

	logistic_sigmoid(std::unique_ptr<expression<A>> &&a) :
		a_(std::move(a).transfer_cast()),
		result_(F(1) / (F(1) + (-((*a_)().eval())).array().exp())) {}

	auto operator()() const {
		return result_.matrix();
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {
		a_.bprop((in.array() * result_ * (1 - result_)).matrix());
	}

private:
	A a_;
	typedef Eigen::Array<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> array_type;
	array_type result_;
};

template<class A>
class softmax : public expression<softmax<A>> {
public:
	typedef typename A::F F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime,
		StorageOrder = A::StorageOrder
	};

private:
	typedef Eigen::Array<F,RowsAtCompileTime,ColsAtCompileTime,StorageOrder> array_;

public:
	softmax(const expression<A> &a) :
			a_(a.cast()) {
		const auto &eval_a = a_().eval().array();
		array_ t = (eval_a.colwise() - eval_a.rowwise().maxCoeff()).exp();
		result_ = t.colwise() / t.rowwise().sum();
	}

	auto operator()() const {
		return result_.matrix();
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {
		a_.bprop((in.array() * result_ * (1 - result_)).matrix());
	}

private:
	const A &a_;
	array_ result_;
};
*/

} // namespace expr

using expr::expression_ptr;
using expr::derived_ptr;

template<class A>
derived_ptr<expr::output_matrix<A>>
eval(expression_ptr<derived_ptr<A>> &&a) {
	return derived_ptr<expr::output_matrix<A>>(
		std::make_unique<expr::output_matrix<A>>(std::move(a).transfer_cast()));
	//return expr::detail::make_derived<expr::output_matrix<A>>(std::move(a));
}

template<class A>
derived_ptr<expr::input_matrix<A>>
input_matrix(const Eigen::MatrixBase<A> &mat) {
	return derived_ptr<expr::input_matrix<A>>(std::make_unique<expr::input_matrix<A>>(mat.derived()));
	//return expr::detail::make_derived<expr::input_matrix<A>>(mat.derived());
}

template<class A,class B>
derived_ptr<expr::weight_matrix<A,B>>
weight_matrix(const Eigen::MatrixBase<A> &mat, const Eigen::MatrixBase<B> &grad) {
	// const_cast according to Eigen manual
	return derived_ptr<expr::weight_matrix<A,B>>(
		std::make_unique<expr::weight_matrix<A,B>>(mat, const_cast<B &>(grad.derived())));
	//return expr::detail::make_derived<expr::weight_matrix<A,B>>(mat, const_cast<B &>(grad.derived()));
}

template<class A,class B>
std::enable_if_t<B::RowsAtCompileTime==1,derived_ptr<expr::rowwise_add<A,B>>>
operator+(expression_ptr<derived_ptr<A>> &&a, expression_ptr<derived_ptr<B>> &&b) {
	return derived_ptr<expr::rowwise_add<A,B>>(
		std::make_unique<expr::rowwise_add<A,B>>(std::move(a).transfer_cast(), std::move(b).transfer_cast()));
	//return expr::detail::make_derived<expr::rowwise_add<A,B>>(std::move(a), std::move(b));
}

template<class A,class B>
derived_ptr<expr::matmul<A,B>>
operator*(expression_ptr<derived_ptr<A>> &&a, expression_ptr<derived_ptr<B>> &&b) {
	return derived_ptr<expr::matmul<A,B>>(
		std::make_unique<expr::matmul<A,B>>(std::move(a).transfer_cast(), std::move(b).transfer_cast()));
	//return expr::detail::make_derived<expr::matmul<A,B>>(std::move(a), std::move(b));
}

/*
template<class A>
derived_ptr<expr::logistic_sigmoid<A>>
logistic_sigmoid(expression_ptr<A> &a) {
	return expr::detail::make_derived<expr::logistic_sigmoid<A>>(std::move(a));
}

template<class A>
expr::softmax<A>
softmax(const expression<A> &a) {
	return expr::softmax<A>(a);
}
*/

} // namespace netops

#endif

#ifndef NNET_NETOPS_H
#define NNET_NETOPS_H

#include <type_traits>
#include <Eigen/Core>

namespace netops {

template<class Derived>
struct expression {
	using typename Derived::F;
	enum {
		RowsAtCompileTime = Derived::RowsAtCompileTime,
		ColsAtCompileTime = Derived::ColsAtCompileTime
	};

	const Derived &cast() const {
		return static_cast<const Derived&>(*this);
	}
};

template<class A>
class input_matrix : public expression<input_matrix> {
public:
	typedef typename A::Scalar F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime
	};

	input_matrix(const A &mat) : mat_(mat) {}

	const A &operator()() const {
		return mat_;
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {}

private:
	const A &mat_;
};

template<class A>
class weight_matrix : public expression<weight_matrix> {
public:
	typedef typename A::Scalar F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime
	};

	weight_matrix(const A &weights, A &gradients) :
			weights_(weights), gradients_(gradients) {
		gradients_.setZero();
	}

	const A &operator()() const {
		return weights_;
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {
		gradients_ += in;
	}

private:
	const A &weights_;
	A &gradients_;
};

template<class A,class B,class Scalar>
class rowwise_add : public expression<rowwise_add> {
public:
	using typename A::F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime
	};

	rowwise_add(const expression<A> &a, const expression<B> &b) :
		a_(a.cast()), b_(b.cast()) {}

	auto operator()() const {
		return a_().rowwise() + b_();
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {
		const auto &eval_in = in.eval();
		a_.bprop(eval_in);
		b_.bprop(eval_in.colwise().sum());
	}

private:
	const A &a_;
	const B &b_;
};

template<class A,class B>
std::enable_if_t<B::RowsAtCompileTime==1,rowwise_add<A,B>>
operator+(const expression<A> &a, const expression<B> &b) {
	return rowwise_add<A,B>(a, b);
}

template<class A,class B>
class matmul : public expression<matmul> {
public:
	using typename A::F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = B::ColsAtCompileTime
	};

	matmul(const expression<A> &a, const expression<B> &b) :
		a_(a.cast()), b_(b.cast()), result_(a_ * b_) {}

	auto operator()() const {
		return result_;
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {
		const auto &eval_in = in.eval();
		a_.bprop(eval_in * b_.transpose());
		b_.bprop(a_.transpose() * eval_in);
	}

private:
	const A &a_;
	const B &b_;
	matrix<F> result_;
};

template<class A,class B>
matmul<A,B>
operator*(const expression<A> &a, const expression<B> &b) {
	return matmul<A,B>(a, b);
}

template<class A>
class logistic_sigmoid_xpr : public expression<logistic_sigmoid_xpr> {
public:
	using typename A::F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime
	};

	logistic_sigmoid_xpr(const expression<A> &a) :
		a_(a.cast()),
		result_(F(1) / (F(1) + (-a()).array().exp()) {}

	auto operator()() const {
		return result_.matrix();
	}

	template<class Derived>
	void bprop(const Eigen::MatrixBase<Derived> &in) const {
		a_.bprop((in.array() * result_ * (1 - result_)).matrix());
	}

private:
	const A &a_;
	array<F> result_;
};

template<class A>
logistic_sigmoid_xpr<A>
logistic_sigmoid(const expression<A> &a) {
	return logistic_sigmoid_xpr<A>(a);
}

template<class A>
class softmax : public expression<softmax> {
public:
	using typename A::F;
	enum {
		RowsAtCompileTime = A::RowsAtCompileTime,
		ColsAtCompileTime = A::ColsAtCompileTime
	};

	softmax(const expression<A> &a) :
			a_(a.cast()) {
		const auto &eval_a = a().eval().array();
		array<F> t = (eval_a.colwise() - eval_a.rowwise().maxCoeff()).exp();
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
	array<F> result_;
};
template<class A>
softmax_xpr<A>
softmax(const expression<A> &a) {
	return softmax_xpr<A>(a);
}

} // namespace netops

#endif

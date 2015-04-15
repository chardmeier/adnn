#include <Eigen/Core>

namespace netops {

template<class Derived>
struct expression {
	using typename Derived::F;

	const Derived &cast() const {
		return static_cast<const Derived&>(*this);
	}
};

template<class A,class B,class Scalar>
class rowwise_add : public expression<rowwise_add> {
public:
	typedef Scalar F;

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

template<class A,class B,class Scalar>
class matmul : public expression<matmul> {
public:
	typedef Scalar F;

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

template<class A,class Scalar>
class logistic_sigmoid : public expression<logistic_sigmoid> {
public:
	typedef Scalar F;

	logistic_sigmoid(const expression<A> &a) :
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

template<class A,class Scalar>
class softmax : public expression<softmax> {
public:
	typedef Scalar F;

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

} // namespace netops

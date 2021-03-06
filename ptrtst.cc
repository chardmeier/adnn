#include <iostream>
#include <memory>

#include <Eigen/Core>

typedef Eigen::MatrixXi val;

// expression_ptr and derived_ptr: contain unique pointers
// to the actual expression objects

template<class Derived>
struct expression_ptr {
	Derived &&transfer_cast() && {
		return std::move(static_cast<Derived &&>(*this));
	}
};

template<class A>
struct derived_ptr : public expression_ptr<derived_ptr<A>> {
	derived_ptr(std::unique_ptr<A> &&p) : ptr_(std::move(p)) {}
	derived_ptr(derived_ptr<A> &&o) = default;

	auto operator()() const {
		return (*ptr_)();
	}

	template<class F>
	auto operator()(F &&f) const {
		return (*ptr_)(std::forward<F>(f));
	}

private:
	std::unique_ptr<A> ptr_;
};

template<class A,class B,class F>
auto binary_cont(const derived_ptr<A> &a_, const derived_ptr<B> &b_, const F &&f) {
	return a_([&b_, f = std::forward<const F>(f)] (auto &&a) {
		return b_([a = std::forward<decltype(a)>(a), f = std::forward<const F>(f)] (auto &&b) {
			return std::forward<const F>(f)(std::forward<decltype(a)>(a), std::forward<decltype(b)>(b));
		});
	});
}

// value_xpr, product_xpr and output_xpr: expression templates
// doing the actual work

template<class A>
struct value_xpr {
	value_xpr(const A &v) : value_(v) {}

	template<class F>
	auto operator()(F &&f) const {
		return std::forward<F>(f)(value_);
	}

private:
	const A &value_;
};

template<class A,class B>
struct product_xpr {
	product_xpr(expression_ptr<derived_ptr<A>> &&a, expression_ptr<derived_ptr<B>> &&b) :
		a_(std::move(a).transfer_cast()), b_(std::move(b).transfer_cast()) {
	}

	template<class F>
	auto operator()(F &&f) const {
		return binary_cont(a_, b_,
			[f = std::forward<F>(f)] (auto &&a, auto &&b) {
				return f(std::forward<decltype(a)>(a) * std::forward<decltype(b)>(b));
			});
	}

private:
	derived_ptr<A> a_;
	derived_ptr<B> b_;
};

template<class A>
struct output_xpr {
	output_xpr(expression_ptr<derived_ptr<A>> &&a) :
			a_(std::move(a).transfer_cast()) {
		a_([this] (auto &&x) { this->result_ = x; });
	}

	const val &operator()() const {
		return result_;
	}

private:
	derived_ptr<A> a_;
	val result_;
};

// helper functions to create the expressions

template<class A>
derived_ptr<value_xpr<A>> input(const A &a) {
	return derived_ptr<value_xpr<A>>(std::make_unique<value_xpr<A>>(a));
}

template<class A,class B>
derived_ptr<product_xpr<A,B>> operator*(expression_ptr<derived_ptr<A>> &&a, expression_ptr<derived_ptr<B>> &&b) {
	return derived_ptr<product_xpr<A,B>>(std::make_unique<product_xpr<A,B>>(std::move(a).transfer_cast(), std::move(b).transfer_cast()));
}

template<class A>
derived_ptr<output_xpr<A>> eval(expression_ptr<derived_ptr<A>> &&a) {
	return derived_ptr<output_xpr<A>>(std::make_unique<output_xpr<A>>(std::move(a).transfer_cast()));
}

int main() {
	Eigen::MatrixXi mat(2, 2);
	mat << 1, 1, 0, 1;
	val one(mat), two(mat), three(mat);
	auto xpr = eval(input(one) * input(two) * input(one) * input(two));
	std::cout << xpr() << std::endl;
	return 0;
}

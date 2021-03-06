#ifndef NNET_NNET_H
#define NNET_NNET_H

#include <chrono>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>

#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/include/advance.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/begin.hpp>
#include <boost/fusion/include/cons.hpp>
#include <boost/fusion/include/deref.hpp>
#include <boost/fusion/include/end.hpp>
#include <boost/fusion/include/flatten.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/fusion/include/make_cons.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/include/transform.hpp>

#include <Eigen/Core>

namespace nnet {

template<class F,int Rows = Eigen::Dynamic,int Cols = Eigen::Dynamic>
using std_matrix = Eigen::Matrix<F,Rows,Cols,Eigen::RowMajor>;

template<class F,int Rows = Eigen::Dynamic,int Cols = Eigen::Dynamic>
using std_array = Eigen::Array<F,Rows,Cols,Eigen::RowMajor>;

// for debugging only
template<class E>
void show_type(E t) {
	t.raise_type_error__();
}

// we define our own functors for exp and log because they
// must be specialised for adept::aReal
template<class F>
struct exp_functor {
	typedef F return_type;
	F operator()(const F &x) const {
		using std::exp;
		return exp(x);
	}
};

template<class F>
struct log_functor {
	typedef F return_type;
	F operator()(const F &x) const {
		using std::log;
		return log(x);
	}
};

// a printable representation of the value
// (will be specialised for AD types)
template<class C>
struct value_functor {
	typedef const C &result_type;
	result_type operator()(const C &in) const {
		return in;
	}
};

struct sigmoid {
	template<class Derived>
	struct functor {
		typedef Eigen::MatrixBase<typename std::remove_const<typename std::remove_reference<Derived>::type>::type> matrix_base;
		std_matrix<typename matrix_base::Scalar> operator()(const matrix_base &x) const {
			typedef typename matrix_base::Scalar F;
			return (F(1) / (F(1) + (-x).array().unaryExpr(exp_functor<F>()))).matrix();
		}
		//typedef decltype(DECLVAL(functor<Derived> )()(DECLVAL(matrix_base)())) result_type;
	};
};

struct softmax {
	template<class Derived>
	struct functor {
		typedef Eigen::MatrixBase<typename std::remove_const<typename std::remove_reference<Derived>::type>::type> matrix_base;
		std_matrix<typename matrix_base::Scalar> operator()(const matrix_base &x) const {
			typedef typename matrix_base::Scalar F;
			std_array<F> a = (x.array().colwise() - x.array().rowwise().maxCoeff()).unaryExpr(exp_functor<F>());
			return (a.colwise() / a.rowwise().sum()).matrix();
		}
		//typedef decltype(DECLVAL(functor<Derived> )()(DECLVAL(matrix_base)())) result_type;
	};
};

template<class Activation,class Matrix>
auto invoke_activation(const Matrix &mat) {
	typename Activation::template functor<Matrix> fn;
	return fn(mat);
}

template<class Derived>
struct spec {
	const Derived &cast() const {
		return static_cast<const Derived &>(*this);
	}

	std::size_t rows() const {
		return cast().rows();
	}

	std::size_t cols() const {
		return cast().cols();
	}

	std::size_t size() const {
		return rows() * cols();
	}
};

template<class F,int TStorageOrder = Eigen::RowMajor>
struct mat_size : public spec<mat_size<F,TStorageOrder>> {
	enum {
		RowsAtCompileTime = Eigen::Dynamic,
		ColsAtCompileTime = Eigen::Dynamic,
		StorageOrder = TStorageOrder
	};
	typedef F float_type;

	mat_size() {}

	mat_size(std::size_t r, std::size_t c) :
		rows_(r), cols_(c) {}

	std::size_t rows() const {
		return rows_;
	}

	std::size_t cols() const {
		return cols_;
	}

	bool operator==(const mat_size &o) const {
		return rows_ == o.rows_ && cols_ == o.cols_;
	}

private:
	std::size_t rows_;
	std::size_t cols_;
};

template<class F,int TStorageOrder = Eigen::RowMajor>
struct vec_size : public spec<vec_size<F,TStorageOrder>> {
	enum {
		RowsAtCompileTime = 1,
		ColsAtCompileTime = Eigen::Dynamic,
		StorageOrder = TStorageOrder
	};
	typedef F float_type;

	vec_size() {}

	vec_size(std::size_t c) :
		cols_(c) {}

	std::size_t rows() const {
		return 1;
	}

	std::size_t cols() const {
		return cols_;
	}

	bool operator==(const vec_size &o) const {
		return cols_ == o.cols_;
	}

private:
	std::size_t cols_;
};

// For configuration data that isn't a matrix.
// rows(), cols() and size() will just fail on this type of data.
template<class Derived>
struct config_data : public spec<config_data<Derived>> {};

template<class FF,class Spec,class Array = std_array<FF> >
class weights;

namespace detail {

constexpr std::size_t align_round(std::size_t size) {
	const std::size_t ALIGNMENT = 32; // alignment in number of floats (a power of two)

	const std::size_t MASK = ALIGNMENT - 1;
	if((size & MASK) == 0)
		return size;
	else
		return (size & ~MASK) + ALIGNMENT;
}

template<class FF>
struct create_weight_maps {
/*
	template<class List>
	auto operator()(const std::pair<FF*,List> &s, const mat_size &e) const;

	template<class List,class List2>
	auto operator()(const std::pair<FF*,List> &s, const List2 &e) const;
*/
	template<class Sequence>
	auto process_sequence(const Sequence &seq, FF *data) const;

	template<class It>
	auto process_sequence(const It &it1, const It &it2, FF *data) const;

	template<class It1,class It2>
	auto process_sequence(const It1 &it1, const It2 &it2, FF *data) const;

	template<class Type>
	auto process_element(const spec<Type> &e, FF *data) const;

	template<class Sequence>
	auto process_element(const Sequence &s, FF *data, std::enable_if_t<boost::fusion::traits::is_sequence<Sequence>::type::value>* = nullptr) const;
};

template<typename>
struct is_eigen_map : std::false_type {};

template<class P,int O,class S>
struct is_eigen_map<Eigen::Map<P,O,S> > : std::true_type {};

} // namespace detail

template<class FF,class Spec,class Array>
class weights {

private:
	template<typename,typename,typename> friend class weights;

	Spec spec_;
	Array data_;

public:
	typedef decltype(detail::create_weight_maps<FF>().process_sequence(spec_, data_.data())) map_type;

private:
	map_type mat_;

public:
	typedef FF float_type;
	typedef Spec spec_type;

	weights(const spec_type &spec) :
			spec_(spec),
			data_(boost::fusion::accumulate(boost::fusion::flatten(spec_), std::size_t(0), [] (std::size_t s, const auto &spec) { return s + detail::align_round(spec.size()); }), 1),
			mat_(detail::create_weight_maps<FF>().process_sequence(spec_, data_.data())) {}

	// Eigen::Map has no copy constructor
	template<class P,int O,class S,class A1 = Array>
	weights(const spec_type &spec, Eigen::Map<P,O,S> &data,
		typename std::enable_if<detail::is_eigen_map<A1>::value>::type* = nullptr) :
			spec_(spec),
			data_(data.data(), data.rows(), data.cols()),
			mat_(detail::create_weight_maps<FF>().process_sequence(spec_, data_.data())) {}

	template<class Derived,class A1 = Array>
	weights(const spec_type &spec, const Eigen::ArrayBase<Derived> &data,
		typename std::enable_if<!detail::is_eigen_map<A1>::value>::type* = nullptr) :
			spec_(spec),
			data_(data),
			mat_(detail::create_weight_maps<FF>().process_sequence(spec_, data_.data())) {}

	weights(const spec_type &spec, const FF &value) :
			spec_(spec),
			data_(boost::fusion::accumulate(boost::fusion::flatten(spec_), std::size_t(0), [] (std::size_t s, const auto &spec) { return s + detail::align_round(spec.size()); }), 1),
			mat_(detail::create_weight_maps<FF>().process_sequence(spec_, data_.data())) {
		data_.setConstant(value);
	}

	template<class FFF,class OSpec,class OArray>
	weights(const weights<FFF,OSpec,OArray> &o) :
			spec_(o.spec_),
			data_(o.data_.template cast<FF>()),
			mat_(detail::create_weight_maps<FF>().process_sequence(spec_, data_.data())) {}

	weights(const weights<FF,Spec,Array> &o) :
			spec_(o.spec_),
			data_(o.data_),
			mat_(detail::create_weight_maps<FF>().process_sequence(spec_, data_.data())) {}

	template<class OtherArray>
	weights<FF,Spec,Array> &operator=(const weights<FF,Spec,OtherArray> &o) {
		assert(spec_ == o.spec_); // assigning weight structures of different shape or size isn't supported
		std::copy(o.data_.data(), o.data_.data() + data_.size(), data_.data());
		return *this;
	}

	weights<FF,Spec,Array> &operator=(const weights<FF,Spec,Array> &o) {
		assert(spec_ == o.spec_); // assigning weight structures of different shape or size isn't supported
		std::copy(o.data_.data(), o.data_.data() + data_.size(), data_.data());
		return *this;
	}

	template<class Functor>
	auto transform(const Functor &f) const {
		typedef decltype(f(FF(0))) FFF;
		return weights<FFF,Spec,std_array<FFF> >(spec_, data_.unaryExpr(f));
	}

	Array &array() {
		return data_;
	}

	const Array &array() const {
		return data_;
	}

	auto &sequence() {
		return mat_;
	}

	const auto &sequence() const {
		return mat_;
	}

	template<int N>
	auto &at() {
		using namespace boost::fusion;
		return deref(advance_c<N>(begin(mat_)));
	}

	template<int N>
	const auto &at() const {
		using namespace boost::fusion;
		return deref(advance_c<N>(begin(mat_)));
	}

	void init_normal(float_type stddev) {
		std::random_device rd;
		std::mt19937 rgen(rd());
		std::normal_distribution<float_type> dist(0, stddev);
		std::generate_n(data_.data(), data_.size(), std::bind(dist, rgen));
	}
};

namespace detail {

template<class FF>
template<class Sequence>
auto create_weight_maps<FF>::process_sequence(const Sequence &seq, FF *data) const {
	using namespace boost::fusion;
	return as_vector(process_sequence(begin(seq), end(seq), data).first);
}

template<class FF>
template<class It>
auto create_weight_maps<FF>::process_sequence(const It &it1, const It &it2, FF *data) const {
	return std::make_pair(boost::fusion::nil_(), data);
}

template<class FF>
template<class It1,class It2>
auto create_weight_maps<FF>::process_sequence(const It1 &it1, const It2 &it2, FF *data) const {
	namespace fusion = boost::fusion;
	auto head = process_element(fusion::deref(it1), data);
	auto tail = process_sequence(fusion::next(it1), it2, head.second);
	return std::make_pair(make_cons(head.first, tail.first), tail.second);
}

template<class FF>
template<class Type>
auto create_weight_maps<FF>::process_element(const spec<Type> &e, FF *data) const {
	typedef Eigen::Map<Eigen::Matrix<FF,Type::RowsAtCompileTime,Type::ColsAtCompileTime,Type::StorageOrder>,Eigen::Aligned> map_type;
	FF *newpos = data + align_round(e.size());
	return std::make_pair(map_type(data, e.rows(), e.cols()), newpos);
}


template<class FF>
template<class Sequence>
auto create_weight_maps<FF>::process_element(const Sequence &s, FF *data, std::enable_if_t<boost::fusion::traits::is_sequence<Sequence>::type::value>*) const {
	using namespace boost::fusion;
	auto l = process_sequence(begin(s), end(s), data);
	return std::make_pair(as_vector(l.first), l.second);
}

struct dump_matrix {
	std::ostream &os_;

	dump_matrix(std::ostream &os) : os_(os) {}

	template<class Matrix>
	void operator()(const Matrix &mat) const {
		os_ << "MATRIX " << mat.rows() << ' ' << mat.cols() << '\n' << mat << '\n';
	}
};

struct data_to_spec {
	template<class Data>
	auto operator()(const Data &d, std::enable_if_t<boost::fusion::traits::is_sequence<Data>::type::value>* = nullptr) const;
	template<class Derived>
	auto operator()(const Eigen::EigenBase<Derived> &m, std::enable_if_t<Derived::RowsAtCompileTime == 1>* = nullptr) const;
	template<class Derived>
	auto operator()(const Eigen::EigenBase<Derived> &m, std::enable_if_t<Derived::RowsAtCompileTime != 1>* = nullptr) const;
	template<class Derived>
	auto operator()(const config_data<Derived> &d) const;
};

template<class Data>
auto data_to_spec::operator()(const Data &d, std::enable_if_t<boost::fusion::traits::is_sequence<Data>::type::value>*) const {
	return boost::fusion::as_vector(boost::fusion::transform(d, data_to_spec()));
}

template<class Derived>
auto data_to_spec::operator()(const Eigen::EigenBase<Derived> &m, std::enable_if_t<Derived::RowsAtCompileTime == 1>*) const {
	constexpr int storage_order = Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor;
	return vec_size<typename Derived::Scalar,storage_order>(m.cols());
}

template<class Derived>
auto data_to_spec::operator()(const Eigen::EigenBase<Derived> &m, std::enable_if_t<Derived::RowsAtCompileTime != 1>*) const {
	constexpr int storage_order = Derived::IsRowMajor ? Eigen::RowMajor : Eigen::ColMajor;
	return mat_size<typename Derived::Scalar,storage_order>(m.rows(), m.cols());
}

template<class Derived>
auto data_to_spec::operator()(const config_data<Derived> &d) const {
	return d;
}

} // namespace detail

template<class Data>
auto data_to_spec(const Data &d) {
	return detail::data_to_spec()(d);
}

template<class FF,class Spec,class Array>
std::ostream &operator<<(std::ostream &os, const weights<FF,Spec,Array> &ww) {
	using namespace boost::fusion;
	const auto &flatseq = flatten(ww.sequence());
	os << "WEIGHTS " << size(flatseq) << '\n';
	for_each(flatseq, detail::dump_matrix(os));
	return os;
}

} // namespace nnet

#endif

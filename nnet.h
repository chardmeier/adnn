#ifndef NNET_NNET_H
#define NNET_NNET_H

#include <cmath>
#include <type_traits>
#include <utility>

#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/include/advance.hpp>
#include <boost/fusion/include/begin.hpp>
#include <boost/fusion/include/deref.hpp>
#include <boost/fusion/include/flatten.hpp>
#include <boost/fusion/include/list.hpp>
#include <boost/fusion/include/push_back.hpp>

#include <Eigen/Core>

namespace nnet {

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
		auto operator()(const matrix_base &x) const {
			typedef typename matrix_base::Scalar F;
			return (F(1) / (F(1) + (-x).array().unaryExpr(exp_functor<F>()))).matrix();
		}
		typedef decltype(std::declval<functor<Derived> >()(std::declval<matrix_base>())) result_type;
	};
};

struct softmax {
	template<class Derived>
	struct functor {
		typedef Eigen::MatrixBase<typename std::remove_const<typename std::remove_reference<Derived>::type>::type> matrix_base;
		auto operator()(const matrix_base &x) const {
			typedef typename matrix_base::Scalar F;
			auto a = (x.array() - x.array().rowwise().maxCoeff().replicate(1, x.cols())).unaryExpr(exp_functor<F>());
			return (a / a.rowwise().sum().replicate(1, x.cols())).matrix();
		}
		typedef decltype(std::declval<functor<Derived> >()(std::declval<matrix_base>())) result_type;
	};
};

template<class F,int Rows = Eigen::Dynamic,int Cols = Eigen::Dynamic>
using std_matrix = Eigen::Matrix<F,Rows,Cols>;

template<class F,int Rows = Eigen::Dynamic,int Cols = Eigen::Dynamic>
using std_array = Eigen::Array<F,Rows,Cols>;

struct mat_size {
	std::size_t rows;
	std::size_t cols;

	mat_size(std::size_t r, std::size_t c) :
		rows(r), cols(c) {}

	bool operator==(const mat_size &o) const {
		return rows == o.rows && cols == o.cols;
	}
};

namespace detail {

struct compute_weight_size {
	typedef std::size_t result_type;
	std::size_t operator()(const std::size_t &s, const mat_size &e) const {
		return s + e.rows * e.cols;
	}
};

template<class FF,class Spec,class Array = std_array<FF> >
class weights;

template<class FF,class Spec>
struct create_weight_maps {
	template<class List>
	auto operator()(const std::pair<FF*,List> &s, const mat_size &e) const {
		typedef Eigen::Map<Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic> > map_type;
		FF *newpos = s.first + e.rows * e.cols;
		return std::make_pair(newpos, push_back(s.second, map_type(s.first, e.rows, e.cols)));
	}

	template<class List,class List2>
	auto operator()(const std::pair<FF*,List> &s, const List2 &e) const {
		typedef Eigen::Map<std_array<FF> > array_map_type;
		const auto &sublist = accumulate(e, std::make_pair(s.first, boost::fusion::list<>()), create_weight_maps<FF,Spec>());
		return std::make_pair(sublist.first, weights<FF,Spec,array_map_type>(
			array_map_type(s.first, sublist.first - s.first, 1), push_back(s.second, sublist.second)));
	}
};

} // namespace detail

template<class FF,class Spec,class Array>
class weights {
private:
	template<typename,typename,typename> friend class weights;

	typedef typename boost::fusion::result_of::accumulate<
			Spec,
			std::pair<FF*,boost::fusion::list<> >,
			detail::create_weight_maps<FF,Spec> >::type::second_type
		map_type;

	Spec spec_;
	Array data_;
	map_type mat_;

public:
	typedef FF float_type;
	typedef Spec spec_type;

	weights(const spec_type &spec) :
			spec_(spec),
			data_(boost::fusion::accumulate(boost::fusion::flatten(spec_), std::size_t(0), detail::compute_weight_size()), 1),
			mat_(boost::fusion::accumulate(spec_, std::make_pair(data_.data(), boost::fusion::list<>()), detail::create_weight_maps<FF,Spec>()).second) {}

	template<class Derived>
	weights(const spec_type &spec, const Eigen::ArrayBase<Derived> &data) :
			spec_(spec),
			data_(data),
			mat_(boost::fusion::accumulate(spec_, std::make_pair(data_.data(), boost::fusion::list<>()), detail::create_weight_maps<FF,Spec>()).second) {}

	weights(const spec_type &spec, const FF &value) :
			spec_(spec),
			data_(boost::fusion::accumulate(boost::fusion::flatten(spec_), std::size_t(0), detail::compute_weight_size()), 1),
			mat_(boost::fusion::accumulate(spec_, std::make_pair(data_.data(), boost::fusion::list<>()), detail::create_weight_maps<FF,Spec>()).second) {
		data_.setConstant(value);
	}

	template<class FFF,class OSpec,class OArray>
	weights(const weights<FFF,OSpec,OArray> &o) :
			spec_(o.spec_),
			data_(o.data_.template cast<FF>()),
			mat_(boost::fusion::accumulate(spec_, std::make_pair(data_.data(), boost::fusion::list<>()), detail::create_weight_maps<FF,Spec>()).second) {}

	weights(const weights<FF,Spec,Array> &o) :
			spec_(o.spec_),
			data_(o.data_),
			mat_(boost::fusion::accumulate(spec_, std::make_pair(data_.data(), boost::fusion::list<>()), detail::create_weight_maps<FF,Spec>()).second) {}

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

	auto &array() {
		return data_;
	}

	const auto &array() const {
		return data_;
	}

	template<int N>
	auto at() {
		using namespace boost::fusion;
		return deref(advance_c<N>(begin(mat_)));
	}

	template<int N>
	const auto at() const {
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

} // namespace nnet

#endif

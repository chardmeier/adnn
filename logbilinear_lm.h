#ifndef NNET_LOGBILINEAR_LM_H
#define NNET_LOGBILINEAR_LM_H

#include <algorithm>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <boost/array.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/boost_array.hpp>
#include <boost/fusion/include/copy.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/front.hpp>
#include <boost/fusion/include/list.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/pop_front.hpp>
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/zip.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <Eigen/SparseCore>

#include "nnet.h"

namespace nnet {

template<class Net,class VocMatrix>
class lblm_dataset;

template<class FF>
using sparse_matrix = Eigen::SparseMatrix<FF,Eigen::RowMajor>; // row-major is essential because of the way the matrix is filled.

namespace detail_lblm {

typedef boost::fusion::vector2<mat_size,vec_size> mat_with_bias;

/*
template<int N> struct C_specs;

template<>
struct C_specs<0> {
	typedef boost::fusion::list<> type;
};

template<int N>
struct C_specs {
	using namespace boost::fusion;
	typedef result_of::push_front<C_specs<N-1>::type,mat_with_bias>::type type;
};
*/

template<int Order,class Matrix>
class ngram_data_type {
private:
	typedef boost::array<Matrix,Order> sequence_type;
	sequence_type sequence_;

public:
	typedef typename Matrix::Scalar float_type;

	static const int order = Order;

	ngram_data_type() {}

	template<class Other>
	ngram_data_type(const Other &o) {
		boost::fusion::copy(o, sequence_);
	}

	sequence_type &sequence() {
		return sequence_;
	}

	const sequence_type &sequence() const {
		return sequence_;
	}

	template<int N>
	Matrix &at() {
		return sequence_[N];
	}

	template<int N>
	const Matrix &at() const {
		return sequence_[N];
	}

	std::size_t nitems() const {
		return sequence_[0].rows();
	}

	template<class NewType>
	auto cast() const {
		return boost::fusion::transform(sequence_, [](const Matrix &mat) { return mat.cast<NewType>(); });
	}
};

} // namespace detail_lblm

template<int Order,class F>
class lblm {
private:
	//typedef detail_lblm::C_specs<Order>::type C_specs;
	//typedef boost::fusion::result_of::push_front<C_specs,detail_lblm::mat_with_bias>::type spec_type;
	typedef boost::array<detail_lblm::mat_with_bias,Order> spec_type;
	spec_type spec_;

	std::size_t voc_size_, embed_size_;

public:
	typedef F float_type;

	typedef lblm_dataset<lblm<Order,F>,sparse_matrix<F> > dataset;

	template<class Matrix>
	using input_type = detail_lblm::ngram_data_type<Order,Matrix>;

	template<class FF>
	using basic_input_type = input_type<sparse_matrix<FF> >;

	template<class Matrix>
	using output_type = detail_lblm::ngram_data_type<1,Matrix>;

	template<class FF>
	using basic_output_type = output_type<std_matrix<FF> >;

	template<class FF>
	using weight_type = weights<FF,spec_type,std_array<FF> >;
	
	lblm(size_t vocsize, size_t embedsize) :
			voc_size_(vocsize), embed_size_(embedsize) {
		using boost::fusion::at_c;
		at_c<0>(spec_[0]) = mat_size(vocsize, embedsize);
		at_c<1>(spec_[0]) = vec_size(vocsize);
		for(int i = 1; i < Order; i++) {
			at_c<0>(spec_[i]) = mat_size(embedsize, embedsize);
			at_c<1>(spec_[i]) = vec_size(embedsize);
		}
	}

	template<class FF,class InputMatrix>
	auto operator()(const weight_type<FF> &W, const input_type<InputMatrix> &inp) const;

	const spec_type &spec() const {
		return spec_;
	}
};

namespace detail_lblm {

template<class Embed,class OutMatrix>
class process_lblm {
private:
	const Embed &embed_;
	OutMatrix &out_;

	template<class T,class U>
	using pair = boost::fusion::vector2<T,U>;

public:
	process_lblm(const Embed &embed, OutMatrix &out) :
		embed_(embed), out_(out) {}

	template<class InpMat,class WMat,class BiasMat>
	void operator()(const pair<InpMat,const pair<WMat,BiasMat>& > &mm) const {
		using boost::fusion::at_c;
		const InpMat &inp = at_c<0>(mm);
		const WMat &w = at_c<0>(at_c<1>(mm));
		const BiasMat &b = at_c<1>(at_c<1>(mm));
		out_.noalias() += (inp * embed_ * w).rowwise() + b;
	}
};

} // namespace detail_lblm

template<int Order,class A>
template<class FF,class InputMatrix>
auto lblm<Order,A>::operator()(const weight_type<FF> &w, const input_type<InputMatrix> &inp) const {
	namespace fusion = boost::fusion;
	const auto &wseq = w.sequence();
	const auto &embed = fusion::at_c<0>(fusion::front(wseq));
	const auto &embed_bias = fusion::at_c<1>(fusion::front(wseq));
	std_matrix<FF> out(inp.nitems(), embed_size_);
	out.setZero();
	fusion::for_each(fusion::zip(inp.sequence(), fusion::pop_front(wseq)),
		detail_lblm::process_lblm<decltype(embed),decltype(out)>(embed, out));
	return output_type<std_matrix<FF> >(fusion::make_vector((out * embed.transpose()).rowwise() + embed_bias));
}

struct lblm_energy {
	template<class T1,class T2>
	struct functor {
		typedef typename T1::float_type result_type;
		result_type operator()(const T1 &output, const T2 &targets) const {
			return -(output.template at<0>() * targets.template at<0>().transpose()).sum();
		}
	};
};

template<class Net,class VocMatrix>
class lblm_batch_iterator;

template<class Net,class VocMatrix>
class lblm_dataset {
public:
	typedef unsigned long vocidx_type;
	typedef std::unordered_map<std::string,vocidx_type> vocmap_type;

private:
	typedef Net net_type;

	typedef typename net_type::template input_type<VocMatrix> input_type;
	typedef typename net_type::template output_type<VocMatrix> output_type;

	input_type inputs_;
	output_type targets_;

	vocmap_type vocmap_;

public:
	typedef lblm_batch_iterator<Net,VocMatrix> batch_iterator;

	lblm_dataset() {}

	template<class InputSequence,class TargetSequence>
	lblm_dataset(const InputSequence &inputs, const TargetSequence &targets) :
		inputs_(inputs), targets_(targets) {}

	const input_type &inputs() const {
		return inputs_;
	}

	input_type &inputs() {
		return inputs_;
	}

	const output_type &targets() const {
		return targets_;
	}

	output_type &targets() {
		return targets_;
	}

	vocmap_type &vocmap() {
		return vocmap_;
	}

	const vocmap_type &vocmap() const {
		return vocmap_;
	}

	std::size_t nitems() const {
		return inputs_.nitems();
	}

	batch_iterator batch_begin(std::size_t batchsize) const;
	batch_iterator batch_end() const;

	void shuffle();
	auto subset(std::size_t from, std::size_t to);
	auto subset(std::size_t from, std::size_t to) const;
};

template<class Net,class VocMatrix,class InputSequence,class TargetSequence>
lblm_dataset<Net,VocMatrix>
make_lblm_dataset(InputSequence inputs, TargetSequence targets) {
	return lblm_dataset<Net,VocMatrix>(inputs, targets);
}

namespace detail_lblm {

template<class Net,class VocMatrix>
struct facade {
	typedef decltype(std::declval<lblm_dataset<Net,VocMatrix> >().subset(std::size_t(0),std::size_t(0))) value_type;
	typedef boost::iterator_facade<lblm_batch_iterator<Net,VocMatrix>,
				value_type, boost::forward_traversal_tag, value_type>
		type;
};

} // namespace detail_lblm

template<class Net,class VocMatrix>
class lblm_batch_iterator
	: public detail_lblm::facade<Net,VocMatrix>::type {
public:
	typedef lblm_dataset<Net,VocMatrix> dataset;

	lblm_batch_iterator(const dataset &data, std::size_t batchsize) :
		data_(data), batchsize_(batchsize), pos_(0) {}

	// end iterator
	lblm_batch_iterator(const dataset &data) :
		data_(data), batchsize_(0), pos_(data.nitems()) {}

private:
	friend class boost::iterator_core_access;

	using typename detail_lblm::facade<Net,VocMatrix>::type::value_type;

	const lblm_dataset<Net,VocMatrix> &data_;
	std::size_t batchsize_;
	std::size_t pos_;

	void increment() {
		pos_ += batchsize_;
		// make sure all end iterators compare equal
		if(pos_ > data_.nitems())
			pos_ = data_.nitems();
	}

	bool equal(const lblm_batch_iterator &other) const {
		if(pos_ == other.pos_)
			return true;

		return false;
	}

	const value_type dereference() const {
		return data_.subset(pos_, pos_ + batchsize_);
	}
};

template<class Net,class VocMatrix>
typename lblm_dataset<Net,VocMatrix>::batch_iterator
lblm_dataset<Net,VocMatrix>::batch_begin(std::size_t batchsize) const {
	return batch_iterator(*this, batchsize);
}

template<class Net,class VocMatrix>
typename lblm_dataset<Net,VocMatrix>::batch_iterator
lblm_dataset<Net,VocMatrix>::batch_end() const {
	return batch_iterator(*this);
}

template<class Net,class VocMatrix>
void lblm_dataset<Net,VocMatrix>::shuffle() {
	Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(nitems());
	perm.setIdentity();
	std::random_device rd;
	std::mt19937 rgen(rd());
	std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), rgen);
	boost::fusion::for_each(inputs_.sequence(), [&perm] (VocMatrix &mat) { mat = perm * mat; });
	boost::fusion::for_each(targets_.sequence(), [&perm] (VocMatrix &mat) { mat = perm * mat; });
}

template<class Net,class VocMatrix>
auto lblm_dataset<Net,VocMatrix>::subset(std::size_t from, std::size_t to) {
	to = std::min(to, nitems());
	auto submat = [from,to] (const VocMatrix &mat) { return mat.middleRows(from, to - from); };
	using boost::fusion::transform;
	return make_lblm_dataset<Net,VocMatrix>(transform(inputs_.sequence(), submat), transform(targets_.sequence(), submat));
}

template<class Net,class VocMatrix>
auto lblm_dataset<Net,VocMatrix>::subset(std::size_t from, std::size_t to) const {
	to = std::min(to, nitems());
	auto submat = [from,to] (const VocMatrix &mat) { return mat.middleRows(from, to - from); };
	using boost::fusion::transform;
	return make_lblm_dataset<Net,VocMatrix>(transform(inputs_.sequence(), submat), transform(targets_.sequence(), submat));
}

} // namespace nnet

#endif

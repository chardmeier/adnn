#ifndef NNET_LOGBILINEAR_LM_H
#define NNET_LOGBILINEAR_LM_H

#include <algorithm>
#include <deque>
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
#include <boost/fusion/include/value_at.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/zip.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mpl/vector_c.hpp>

#include <Eigen/SparseCore>

#include "nnet.h"
#include "netops.h"
#include "vocmap.h"

namespace lblm {

namespace fusion = boost::fusion;
namespace mpl = boost::mpl;

template<class InputMatrix,class TargetMatrix>
class lblm_dataset;

template<class FF>
using sparse_matrix = Eigen::SparseMatrix<FF,Eigen::RowMajor>; // row-major is essential because of the way the matrix is filled.

typedef Eigen::Matrix<vocmap::voc_id,Eigen::Dynamic,1> vocidx_vector;

namespace detail_lblm {

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

	template<int N = Order>
	Matrix &matrix(typename std::enable_if<N==1>::type* = nullptr) {
		return sequence_[0];
	}

	template<int N = Order>
	const Matrix &matrix(typename std::enable_if<N==1>::type* = nullptr) const {
		return sequence_[0];
	}

	template<int N = Order>
	auto array(typename std::enable_if<N==1>::type* = nullptr) {
		return sequence_[0].array();
	}

	template<int N = Order>
	auto array(typename std::enable_if<N==1>::type* = nullptr) const {
		return sequence_[0].array();
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

namespace idx {
	typedef mpl::vector1_c<int,0> I_W3;
	typedef mpl::vector1_c<int,1> I_W2;
	typedef mpl::vector1_c<int,2> I_W1;

	typedef mpl::vector1_c<int,0> W_W;
	typedef mpl::vector2_c<int,0,0> W_Wmat;
	typedef mpl::vector1_c<int,1> W_C3;
	typedef mpl::vector1_c<int,2> W_C2;
	typedef mpl::vector1_c<int,3> W_C1;
} // namespace idx

template<class Float,class Spec,class Net>
class lblm {
private:
	typedef Spec spec_type;
	spec_type spec_;
	netops::derived_ptr<Net> net_;

public:
	typedef Float float_type;
	typedef nnet::weights<Float,Spec> weight_type;

	typedef lblm_dataset<vocidx_vector,sparse_matrix<Float> > dataset;

	typedef typename dataset::input_type input_type;
	typedef typename dataset::output_type output_type;

	lblm(Spec &&spec, netops::expression_ptr<netops::derived_ptr<Net>> &&net) :
		spec_(std::move(spec)), net_(std::move(net).transfer_cast()) {}

	template<class InputType>
	auto operator()(const weight_type &W, const InputType &inp) const;

	template<class InputType,class OutputType>
	auto bprop(const InputType &input, const OutputType &targets, const weight_type &weights, weight_type &grads);

	template<class OutputType,class TargetType>
	float_type error(const OutputType &output, const TargetType &targets) const {
		return -(targets.array() * output.array().log()).sum() / output.rows();
	}

	const spec_type &spec() const {
		return spec_;
	}
};

template<class Float>
auto lweights(std::size_t rows, std::size_t cols) {
	return fusion::vector2<nnet::mat_size<Float>,nnet::vec_size<Float>>(
		nnet::mat_size<Float>(rows, cols), nnet::vec_size<Float>(cols));
}

template<class Float,class Inputs>
auto make_lblm(const Inputs &input, std::size_t vocsize, std::size_t embedsize) {
	auto ispec = nnet::data_to_spec(input.sequence());

	auto wspec = fusion::make_vector(
		lweights<Float>(vocsize, embedsize),
		lweights<Float>(embedsize, embedsize),
		lweights<Float>(embedsize, embedsize),
		lweights<Float>(embedsize, embedsize));

	typedef nnet::weights<Float,decltype(wspec)> weights;

	using namespace netops;
	auto &&net = softmax_crossentropy(
		(linear_layer<idx::W_C3>(wspec, linear_layer<idx::W_W>(wspec, input_matrix<idx::I_W3>(ispec))) +
		linear_layer<idx::W_C2>(wspec, linear_layer<idx::W_W>(wspec, input_matrix<idx::I_W2>(ispec))) +
		linear_layer<idx::W_C1>(wspec, linear_layer<idx::W_W>(wspec, input_matrix<idx::I_W1>(ispec))) *
			transpose(weight_matrix<idx::W_Wmat>(wspec))));

	typedef typename std::remove_reference<decltype(net)>::type::expr_type net_type;
	return lblm<Float,decltype(wspec),net_type>(std::move(wspec), std::move(net));
}

template<class Float,class Spec,class Net>
template<class InputType>
auto lblm<Float,Spec,Net>::operator()(const weight_type &weights, const InputType &input) const {
	return net_.fprop(input, weights.sequence());
}

template<class Float,class Spec,class Net>
template<class InputType,class OutputType>
auto lblm<Float,Spec,Net>::bprop(const InputType &input, const OutputType &targets, const weight_type &weights, weight_type &grads) {
	return net_.bprop_loss(targets.matrix(), input, weights.sequence(), grads.sequence());
}

template<class InputMatrix,class TargetMatrix>
class lblm_batch_iterator;

template<class InputMatrix,class TargetMatrix>
class lblm_dataset {
public:
	typedef std::unordered_map<std::string,vocmap::voc_id> vocmap_type;

	typedef detail_lblm::ngram_data_type<3,InputMatrix> input_type;
	typedef detail_lblm::ngram_data_type<1,TargetMatrix> output_type;
	
private:
	input_type inputs_;
	output_type targets_;

	vocmap_type vocmap_;

public:
	typedef lblm_batch_iterator<InputMatrix,TargetMatrix> batch_iterator;

	lblm_dataset() {}

	template<class InputSequence,class TargetSequence>
	lblm_dataset(const InputSequence &inputs, const TargetSequence &targets) :
		inputs_(inputs), targets_(targets) {}

	auto sequence() const {
		return inputs_.sequence();
	}

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

template<class InputSequence,class TargetSequence>
auto make_lblm_dataset(InputSequence inputs, TargetSequence targets) {
	typedef typename fusion::result_of::value_at_c<InputSequence,0>::type InputMatrix;
	typedef typename fusion::result_of::value_at_c<TargetSequence,0>::type TargetMatrix;
	return lblm_dataset<InputMatrix,TargetMatrix>(inputs, targets);
}

template<class Float,bool Extend>
auto load_lblm(const std::string &infile, vocmap::vocmap &voc) {
	const int ORDER = 3;
	const vocmap::voc_id BOS = voc.lookup("<s>", Extend);

	typedef Eigen::Triplet<Float> triplet;
	boost::array<std::vector<triplet>,ORDER> words;

	std::size_t idx = 0;
	std::ifstream in(infile.c_str());
	for(std::string line; getline(in, line); ) {
		std::istringstream ls(line + " </s>");
		std::deque<vocmap::voc_id> ngram(ORDER, BOS);
		for(std::string word; getline(ls, word, ' '); ) {
			ngram.pop_front();
			ngram.push_back(voc.lookup(word, Extend));
			for(int i = 0; i < ORDER; i++)
				words[i].push_back(triplet(idx++, ngram[i], Float(1)));
		}
	}

	typedef Eigen::SparseMatrix<Float,Eigen::RowMajor> wordinput_type;
	boost::array<wordinput_type,ORDER - 1> history;
	for(int i = 0; i < ORDER - 1; i++)
		history[i].setFromTriplets(words[i].begin(), words[i].end());
	wordinput_type target;
	target.setFromTriplets(words.back().begin(), words.back().end());

	// the target matrix must be dense
	Eigen::Matrix<Float,Eigen::Dynamic,Eigen::Dynamic> target_dense(target);
	
	return make_lblm_dataset(history, fusion::make_vector(target_dense));
}

namespace detail_lblm {

template<class InputMatrix,class TargetMatrix>
struct facade {
	typedef decltype(lblm_dataset<InputMatrix,TargetMatrix>().subset(std::size_t(0),std::size_t(0))) value_type;
	typedef boost::iterator_facade<lblm_batch_iterator<InputMatrix,TargetMatrix>,
				value_type, boost::forward_traversal_tag, value_type>
		type;
};

} // namespace detail_lblm

template<class InputMatrix,class TargetMatrix>
class lblm_batch_iterator
	: public detail_lblm::facade<InputMatrix,TargetMatrix>::type {
public:
	typedef lblm_dataset<InputMatrix,TargetMatrix> dataset;

	lblm_batch_iterator(const dataset &data, std::size_t batchsize) :
		data_(data), batchsize_(batchsize), pos_(0) {}

	// end iterator
	lblm_batch_iterator(const dataset &data) :
		data_(data), batchsize_(0), pos_(data.nitems()) {}

private:
	friend class boost::iterator_core_access;

	using typename detail_lblm::facade<InputMatrix,TargetMatrix>::type::value_type;

	const lblm_dataset<InputMatrix,TargetMatrix> &data_;
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

template<class InputMatrix,class TargetMatrix>
typename lblm_dataset<InputMatrix,TargetMatrix>::batch_iterator
lblm_dataset<InputMatrix,TargetMatrix>::batch_begin(std::size_t batchsize) const {
	return batch_iterator(*this, batchsize);
}

template<class InputMatrix,class TargetMatrix>
typename lblm_dataset<InputMatrix,TargetMatrix>::batch_iterator
lblm_dataset<InputMatrix,TargetMatrix>::batch_end() const {
	return batch_iterator(*this);
}

template<class InputMatrix,class TargetMatrix>
void lblm_dataset<InputMatrix,TargetMatrix>::shuffle() {
	Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(nitems());
	perm.setIdentity();
	std::random_device rd;
	std::mt19937 rgen(rd());
	std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), rgen);
	boost::fusion::for_each(inputs_.sequence(), [&perm] (InputMatrix &mat) { mat = perm * mat; });
	boost::fusion::for_each(targets_.sequence(), [&perm] (TargetMatrix &mat) { mat = perm * mat; });
}

namespace detail_lblm {

struct submatrix_functor {
	std::size_t from_, N_;

	submatrix_functor(std::size_t f, std::size_t t) : from_(f), N_(t - f) {}

	template<class Derived>
	auto operator()(const Eigen::MatrixBase<Derived> &inpmat) const {
		return inpmat.middleRows(from_, N_).eval();
	}

	template<class Derived>
	auto operator()(const Eigen::SparseMatrixBase<Derived> &inpmat) const {
		return inpmat.middleRows(from_, N_).eval();
	}
};

} // namespace detail_lblm

template<class InputMatrix,class TargetMatrix>
auto lblm_dataset<InputMatrix,TargetMatrix>::subset(std::size_t from, std::size_t to) {
	to = std::min(to, nitems());
	detail_lblm::submatrix_functor submat(from, to);
	using boost::fusion::transform;
	return make_lblm_dataset(transform(inputs_.sequence(), submat), transform(targets_.sequence(), submat));
}

template<class InputMatrix,class TargetMatrix>
auto lblm_dataset<InputMatrix,TargetMatrix>::subset(std::size_t from, std::size_t to) const {
	to = std::min(to, nitems());
	detail_lblm::submatrix_functor submat(from, to);
	using boost::fusion::transform;
	return make_lblm_dataset(transform(inputs_.sequence(), submat), transform(targets_.sequence(), submat));
}

} // namespace nnet

#endif

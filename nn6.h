#ifndef NNET_NN6_H
#define NNET_NN6_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/array.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/back.hpp>
#include <boost/fusion/include/boost_array.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/pop_front.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/vector_c.hpp>

#include "nnet.h"
#include "netops.h"

namespace nn6 {

namespace fusion = boost::fusion;
namespace mpl = boost::mpl;

typedef unsigned long voc_id;

namespace detail {

template<bool TrainingMode>
struct nn6_mode : public nnet::config_data<nn6_mode<TrainingMode>> {
	typedef mpl::bool_<TrainingMode> training_mode;
};

template<class Dataset> class batch_iterator;

} // namespace detail

template<class InputSeq,class Targets>
class nn6_dataset {
public:
	typedef InputSeq input_seq;
	typedef detail::batch_iterator<nn6_dataset<InputSeq,Targets>> batch_iterator;

/*
	typedef fusion::vector2<
			fusion::vector3<sparse_matrix,matrix,int_vector>,
			boost::array<sparse_matrix,7>> input_seq;
*/

	nn6_dataset(int nlink, input_seq &&input, Targets &&targets) :
		nlink_(nlink), input_(std::forward<InputSeq>(input)), targets_(std::forward<Targets>(targets)) {}

	const input_seq &sequence() const {
		return input_;
	}

	auto spec() const {
		return nnet::data_to_spec(input_);
	}

	int nlink() const {
		return nlink_;
	}

	std::size_t nitems() const {
		return targets_.rows();
	}

	std::size_t nclasses() const {
		return targets_.cols();
	}

	auto subset(std::size_t from, std::size_t to) const;

	const input_seq &input() const {
		return input_;
	}

	const Targets &targets() const {
		return targets_;
	}

	batch_iterator batch_begin(std::size_t batchsize) const;
	batch_iterator batch_end() const;

private:
	int nlink_;
	input_seq input_;
	Targets targets_;
};

namespace detail {

template<class Dataset>
using iterator_value_type =
	decltype(std::declval<Dataset>().subset(std::size_t(0),std::size_t(0)));

template<class Dataset>
class batch_iterator
	: public boost::iterator_facade<
			batch_iterator<Dataset>,
			iterator_value_type<Dataset>,
			boost::forward_traversal_tag,
			iterator_value_type<Dataset>> {
public:
	typedef Dataset dataset;

	batch_iterator(const dataset &data, std::size_t batchsize) :
		data_(data), batchsize_(batchsize), pos_(0) {}

	// end iterator
	batch_iterator(const dataset &data) :
		data_(data), batchsize_(0), pos_(data.nitems()) {}

private:
	friend class boost::iterator_core_access;

	const Dataset &data_;
	std::size_t batchsize_;
	std::size_t pos_;

	void increment() {
		pos_ += batchsize_;
		// make sure all end iterators compare equal
		if(pos_ > data_.nitems())
			pos_ = data_.nitems();
	}

	bool equal(const batch_iterator<Dataset> &other) const {
		if(pos_ == other.pos_)
			return true;

		return false;
	}

	const iterator_value_type<Dataset> dereference() const {
		return data_.subset(pos_, pos_ + batchsize_);
	}
};

} // namespace detail

template<class InputSeq,class Targets>
typename nn6_dataset<InputSeq,Targets>::batch_iterator
nn6_dataset<InputSeq,Targets>::batch_begin(std::size_t batchsize) const {
	return batch_iterator(*this, batchsize);
}

template<class InputSeq,class Targets>
typename nn6_dataset<InputSeq,Targets>::batch_iterator
nn6_dataset<InputSeq,Targets>::batch_end() const {
	return batch_iterator(*this);
}

template<class InputSeq,class Targets>
auto make_nn6_dataset(int nlink, InputSeq &&input, Targets &&targets) {
	return nn6_dataset<InputSeq,Targets>(nlink, std::forward<InputSeq>(input), std::forward<Targets>(targets));
}

namespace idx {
	// The indices here must be consistent with load_nn6 and nn6_dataset::subset.

	typedef mpl::vector2_c<int,0,0> I_A;
	typedef mpl::vector2_c<int,0,1> I_T;
	typedef mpl::vector2_c<int,0,2> I_antmap;
	typedef mpl::vector2_c<int,0,3> I_nada;

	typedef mpl::vector1_c<int,1> I_srcctx;
	typedef mpl::vector2_c<int,1,0> I_L3;
	typedef mpl::vector2_c<int,1,1> I_L2;
	typedef mpl::vector2_c<int,1,2> I_L1;
	typedef mpl::vector2_c<int,1,3> I_P;
	typedef mpl::vector2_c<int,1,4> I_R1;
	typedef mpl::vector2_c<int,1,5> I_R2;
	typedef mpl::vector2_c<int,1,6> I_R3;

	typedef mpl::vector1_c<int,2> I_mode;

	typedef mpl::vector1_c<int,0> W_U;
	typedef mpl::vector1_c<int,1> W_V;
	typedef mpl::vector1_c<int,2> W_antembed;
	typedef mpl::vector1_c<int,3> W_srcembed;
	typedef mpl::vector1_c<int,4> W_embhid;
	typedef mpl::vector1_c<int,5> W_hidout;
} // namespace idx

template<class Float,class Spec,class Net>
class nn6 {
private:
	typedef Spec spec_type;
	spec_type spec_;
	netops::derived_ptr<Net> net_;

public:
	typedef Float float_type;
	typedef nnet::weights<Float,Spec> weight_type;

	nn6(Spec &&spec, netops::expression_ptr<netops::derived_ptr<Net>> &&net) :
		spec_(std::move(spec)), net_(std::move(net).transfer_cast()) {}

	auto spec() const {
		return spec_;
	}

	template<class InputType>
	auto operator()(const weight_type &weights, const InputType &inputs);

	template<class InputType,class OutputType>
	void bprop(const InputType &input, const OutputType &targets, const weight_type &weights, weight_type &grads);

	template<class OutputType,class TargetType>
	float_type error(const OutputType &output, const TargetType &targets) const {
		return -(targets.array() * output.array().log()).sum() / output.rows();
	}
};

struct vocmap {
	typedef std::unordered_map<std::string,voc_id> map_type;

	enum { UNKNOWN_WORD = 0 };

	voc_id maxid;
	map_type map;

	vocmap() : maxid(1) {
		map.insert(std::make_pair("<unk>", UNKNOWN_WORD));
	}
};

voc_id voc_lookup(const std::string &word, vocmap &voc, bool extend = false) {
	voc_id id;
	vocmap::map_type::const_iterator it = voc.map.find(word);
	if(it != voc.map.end()) 
		id = it->second;
	else {
		if(extend) {
			id = voc.maxid++;
			voc.map.insert(std::make_pair(word, id));
		} else
			id = vocmap::UNKNOWN_WORD;
	}

	return id;
}

class classmap {
public:
	typedef std::unordered_map<std::string,int> map_type;

	classmap();
	classmap(const std::string &file, bool with_other);

	const map_type &map() const {
		return map_;
	}

	int nclasses() const {
		return nclasses_;
	}

	bool with_other() const {
		return with_other_;
	}

	int other() const {
		return nclasses_ - 1;
	}

private:
	map_type map_;
	int nclasses_;
	bool with_other_;
};

classmap::classmap() {
	//enum { CE, CELA, ELLE, ELLES, IL, ILS, ON, CA, OTHER, NCLASSES };
	enum { CE, CELA, ELLE, ELLES, IL, ILS, ON, OTHER, NCLASSES };

	map_.insert(std::make_pair("ce", CE));
	map_.insert(std::make_pair("c'", CE));
	map_.insert(std::make_pair("cela", CELA));
	map_.insert(std::make_pair("elle", ELLE));
	map_.insert(std::make_pair("elles", ELLES));
	map_.insert(std::make_pair("il", IL));
	map_.insert(std::make_pair("ils", ILS));
	map_.insert(std::make_pair("on", ON));
	//map_.insert(std::make_pair("ça", CA));
	//map_.insert(std::make_pair("ca", CA));
	//map_.insert(std::make_pair("ç'", CA));
	map_.insert(std::make_pair("ça", CELA));
	map_.insert(std::make_pair("ca", CELA));
	map_.insert(std::make_pair("ç'", CELA));

	nclasses_ = NCLASSES;
	with_other_ = true;
}

classmap::classmap(const std::string &file, bool with_other) : nclasses_(0), with_other_(with_other) {
	std::ifstream cls(file.c_str());
	if(!cls.good()) {
		std::cerr << "Error opening class map file: " << file << std::endl;
		throw 0;
	}
	for(std::string line; getline(cls, line); nclasses_++) {
		std::istringstream ss(line.c_str());
		for(std::string pron; ss >> pron; )
			map_.insert(std::make_pair(pron, nclasses_));
	}

	if(with_other)
		nclasses_++;
}

class tagmap {
public:
	typedef std::unordered_set<std::string> tagset_type;

private:
	typedef std::unordered_map<std::string,tagset_type> map_type;
	map_type map_;

public:
	tagmap() {}
	tagmap(const std::string &file);

	const tagset_type &lookup(const std::string &word) const {
		static const tagset_type EMPTY_TAGSET;
		map_type::const_iterator it = map_.find(word);
		if(it != map_.end())
			return it->second;
		else
			return EMPTY_TAGSET;
	}
};

tagmap::tagmap(const std::string &file) {
	std::ifstream tss(file.c_str());
	if(!tss.good()) {
		std::cerr << "Error opening tagset file: " << file << std::endl;
		throw 0;
	}
	for(std::string line; getline(tss, line); ) {
		std::istringstream ss(line.c_str());
		std::string word;
		if(ss >> word) {
			tagset_type tagset;
			std::string tag;
			while(ss >> tag)
				tagset.insert(tag);
			map_.insert(std::make_pair(word, tagset));
		}
	}
}

template<class Float,bool TrainingMode>
auto load_nn6(const std::string &file, const classmap &classes, vocmap &srcvocmap, vocmap &antvocmap, const tagmap &tags, int nlink = -1) {
	typedef Eigen::SparseMatrix<Float,Eigen::RowMajor> sparse_matrix;
	typedef Eigen::Matrix<Float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> matrix; // must be row-major because of resizing!
	typedef Eigen::Matrix<int,Eigen::Dynamic,1> int_vector;

	bool set_nlink;
	if(nlink < 0) {
		set_nlink = true;
		nlink = 0;
	} else
		set_nlink = false;

	std::cerr << "Loading file " << file << std::endl;

	std::ifstream is(file);
	if(!is.good()) {
		std::cerr << "Can't open file." << std::endl;
		throw 0;
	}

	std::size_t nexmpl = 0;
	std::size_t nant = 0;

	std::vector<std::string> nn6_lines;
	for(std::string line; getline(is, line);) {
		if(line.compare(0, 8, "ANAPHORA") == 0)
			nexmpl++;
		else if(line.compare(0, 10, "ANTECEDENT") == 0)
			nant++;
		else if(line.compare(0, 2, "-1") == 0 && set_nlink) {
			std::istringstream ss(line);
			int idx;
			char colon;
			Float val;
			ss >> idx; // get rid of -1 at the beginning
			while(ss >> idx >> colon >> val)
				if(idx > nlink)
					nlink = idx;
		}
		nn6_lines.push_back(line);
	}

	typedef Eigen::Matrix<Float,Eigen::Dynamic,1> vector;
	typedef Eigen::Matrix<voc_id,Eigen::Dynamic,1> vocid_vector;
	typedef Eigen::Matrix<int,Eigen::Dynamic,1> int_vector;

	matrix targets(nexmpl, classes.nclasses());
	vector nada(nexmpl);
	matrix T(nant, nlink);
	int_vector antmap(nexmpl);
	std::vector<Eigen::Triplet<Float>> ant_triplets;
	std::vector<std::vector<Eigen::Triplet<Float>>> srcctx_triplets(7);

	nada.setZero();
	T.setZero();
	antmap.setZero();
	targets.setZero();

	bool skip_example = false;

	std::cerr << "In input file: " << nexmpl << " examples, " << nant << " antecedents.\n";

	std::size_t ex = std::numeric_limits<std::size_t>::max();
	std::size_t ant = std::numeric_limits<std::size_t>::max();
	std::size_t total_ant = std::numeric_limits<std::size_t>::max();
	for(std::size_t i = 0; i < nn6_lines.size(); i++) {
		std::istringstream ss(nn6_lines[i]);
		std::string tag;
		getline(ss, tag, ' ');
		if(tag == "ANAPHORA") {
			if(ex < nexmpl)
				antmap(ex) = ant + 1;

			skip_example = false;
			ex++; // wraps around to zero at first example
			std::string word;
			for(int j = 0; j < 7; j++) {
				getline(ss, word, ' ');
				voc_id v = voc_lookup(word, srcvocmap, TrainingMode);
				srcctx_triplets[j].push_back(Eigen::Triplet<Float>(ex, v, 1));
			}
			ant = std::numeric_limits<std::size_t>::max();
		} else if(tag == "TARGET") {
			std::string word;
			bool found = false;
			while(getline(ss, word, ' ')) {
				boost::to_lower(word);
				classmap::map_type::const_iterator it = classes.map().find(word);
				if(it != classes.map().end()) {
					targets(ex, it->second)++;
					found = true;
				}
			}
			if(!found) {
				if(classes.with_other())
					targets(ex, classes.other()) = 1;
				else {
					skip_example = true;
					// undo everything done by the ANAPHORA branch
					ex--;
					for(int j = 0; j < 7; j++)
						srcctx_triplets[j].pop_back();
				}
			}
		} else {
			if(!skip_example) {
				if(tag == "NADA")
					ss >> nada(ex);
				else if(tag == "ANTECEDENT") {
					total_ant++; // wraps around to zero at first antecedent in file
					ant++; // wraps around to zero at first antecedent per example
					int nwords = std::count(nn6_lines[i].begin(), nn6_lines[i].end(), ' ');
					std::string word;
					while(getline(ss, word, ' ')) {
						voc_id v = voc_lookup(word, antvocmap, TrainingMode);
						ant_triplets.push_back(Eigen::Triplet<Float>(total_ant, v, 1.0 / nwords));
						const tagmap::tagset_type &tagset = tags.lookup(word);
						for(const std::string &t : tagset) {
							const std::string PREFIX("lefff:");
							v = voc_lookup(PREFIX + t, antvocmap, TrainingMode);
							if(v != vocmap::UNKNOWN_WORD)
								ant_triplets.push_back(Eigen::Triplet<Float>(total_ant, v, 1.0 / nwords));
						}
					}
				} else if(tag == "-1") {
					int fidx;
					char colon;
					Float fval;
					while(ss >> fidx >> colon >> fval)
						if(fidx <= nlink)
							T(total_ant, fidx - 1) = fval;
				}
			} else if(tag == "ANTECEDENT")
				nant--; // subtract skipped antecedents from total count
		}
	}

	// set number of antecedent candidates for last example
	antmap(ex) = ant + 1;

	// if there's no OTHER, the total number of examples may be lower than the initial estimate
	if(ex + 1 < nexmpl) {
		nexmpl = ex + 1;
		std::cerr << "Loaded: " << nexmpl << " examples, " << nant << " antecedents.\n";
		nada.conservativeResize(nexmpl);
		antmap.conservativeResize(nexmpl);
		targets.conservativeResize(nexmpl, Eigen::NoChange);
		T.conservativeResize(nant, Eigen::NoChange);
	}

	targets.array().colwise() /= targets.array().rowwise().sum();

	typedef Eigen::SparseMatrix<Float,Eigen::RowMajor> wordinput_type;

	wordinput_type A(nant, antvocmap.map.size());
	A.setFromTriplets(ant_triplets.begin(), ant_triplets.end());

	wordinput_type L3(nexmpl, srcvocmap.map.size());
	L3.setFromTriplets(srcctx_triplets[0].begin(), srcctx_triplets[0].end());

	wordinput_type L2(nexmpl, srcvocmap.map.size());
	L2.setFromTriplets(srcctx_triplets[1].begin(), srcctx_triplets[1].end());

	wordinput_type L1(nexmpl, srcvocmap.map.size());
	L1.setFromTriplets(srcctx_triplets[2].begin(), srcctx_triplets[2].end());

	wordinput_type P(nexmpl, srcvocmap.map.size());
	P.setFromTriplets(srcctx_triplets[3].begin(), srcctx_triplets[3].end());

	wordinput_type R1(nexmpl, srcvocmap.map.size());
	R1.setFromTriplets(srcctx_triplets[4].begin(), srcctx_triplets[4].end());

	wordinput_type R2(nexmpl, srcvocmap.map.size());
	R2.setFromTriplets(srcctx_triplets[5].begin(), srcctx_triplets[5].end());

	wordinput_type R3(nexmpl, srcvocmap.map.size());
	R3.setFromTriplets(srcctx_triplets[6].begin(), srcctx_triplets[6].end());

	return make_nn6_dataset(nlink,
		fusion::make_vector(fusion::make_vector(A, T, antmap, nada),
			boost::array<wordinput_type,7>({ L3, L2, L1, P, R1, R2, R3 }),
			detail::nn6_mode<TrainingMode>()),
		std::move(targets));
}

template<class Dataset>
void dump_nn6_dataset(const std::string &outstem, const Dataset &dataset, const vocmap &srcvocmap, const vocmap &antvocmap) {
	Eigen::IOFormat dense_format(Eigen::FullPrecision, Eigen::DontAlignCols, " ", "\n", "", "", "", "\n");

	// srcvoc
	std::size_t srcvocsize = srcvocmap.map.size();
	std::vector<std::string> voc(srcvocsize);
	for(vocmap::map_type::const_iterator it = srcvocmap.map.begin(); it != srcvocmap.map.end(); ++it)
		voc[it->second] = it->first;
	std::ofstream srcvoc_os((outstem + ".srcvoc").c_str());
	for(std::size_t i = 0; i < srcvocsize; i++)
		srcvoc_os << voc[i] << '\n';
	srcvoc_os.close();
	
	// tgtvoc
	std::size_t antvocsize = antvocmap.map.size();
	voc.resize(antvocsize);
	for(vocmap::map_type::const_iterator it = antvocmap.map.begin(); it != antvocmap.map.end(); ++it)
		voc[it->second] = it->first;
	std::ofstream antvoc_os((outstem + ".tgtvoc").c_str());
	for(std::size_t i = 0; i < antvocsize; i++)
		antvoc_os << voc[i] << '\n';
	antvoc_os.close();
	
	// srcfeat
	std::ofstream srcfeat_os((outstem + ".srcfeat").c_str());
	const auto &src_array = netops::at_spec<idx::I_srcctx>(dataset.sequence());
	typedef typename std::remove_reference<decltype(src_array)>::type::value_type wordinput_type;
	for(std::size_t i = 0; i < dataset.nitems(); i++)
		for(std::size_t j = 0, voc_offset = 1; j < src_array.size(); j++, voc_offset += srcvocsize)
			for(typename wordinput_type::InnerIterator it(src_array[j], i); it; ++it)
				srcfeat_os << (i + 1) << ' ' << (it.col() + voc_offset) << ' ' << it.value() << '\n';
	srcfeat_os.close();

	// antfeat
	std::ofstream antfeat_os((outstem + ".antfeat").c_str());
	const auto &A = netops::at_spec<idx::I_A>(dataset.sequence());
	for(int i = 0; i < A.rows(); i++)
		for(typename wordinput_type::InnerIterator it(A, i); it; ++it)
			antfeat_os << (i + 1) << ' ' << (it.col() + 1) << ' ' << it.value() << '\n';
	antfeat_os.close();

	// targets
	std::ofstream targets_os((outstem + ".targets").c_str());
	targets_os << dataset.targets().format(dense_format);
	targets_os.close();
	
	// antmap
	std::ofstream antmap_os((outstem + ".antmap").c_str());
	const auto &antmap = netops::at_spec<idx::I_antmap>(dataset.sequence());
	for(std::size_t i = 0; i < dataset.nitems(); i++)
		for(int j = 0; j < antmap(i); j++)
			antmap_os << (i + 1) << '\n';
	antmap_os.close();
	
	// linkfeat
	std::ofstream linkfeat_os((outstem + ".linkfeat").c_str());
	const auto &T = netops::at_spec<idx::I_T>(dataset.sequence());
	for(int i = 0; i < T.rows(); i++)
		for(int j = 0; j < T.cols(); j++)
			if(T(i,j) != 0)
				linkfeat_os << (i+1) << ' ' << (j+1) << ' ' << T(i,j) << '\n';
	linkfeat_os.close();
	
	// nada
	std::ofstream nada_os((outstem + ".nada").c_str());
	const auto &nada = netops::at_spec<idx::I_nada>(dataset.sequence());
	nada_os << nada.format(dense_format);
	nada_os.close();
}

template<class Float>
auto lweights(std::size_t rows, std::size_t cols) {
	return fusion::vector2<nnet::mat_size<Float>,nnet::vec_size<Float>>(
		nnet::mat_size<Float>(rows, cols), nnet::vec_size<Float>(cols));
}

template<class Float,class Inputs>
auto make_nn6(const Inputs &input,
		std::size_t size_U, std::size_t size_antembed, std::size_t size_srcembed, std::size_t size_hidden,
		std::size_t size_output, Float dropout_src) {
	auto ispec = nnet::data_to_spec(input);
	int size_T = netops::at_spec<idx::I_T>(ispec).cols();
	int size_ant = netops::at_spec<idx::I_A>(ispec).cols();
	int size_src = netops::at_spec<idx::I_L1>(ispec).cols();

	auto wspec = fusion::make_vector(
			lweights<Float>(size_T, size_U),
			lweights<Float>(size_U, 1),
			lweights<Float>(size_ant, size_antembed),
			lweights<Float>(size_src, size_srcembed),
			lweights<Float>(7 * size_srcembed + size_antembed + 1, size_hidden),
			lweights<Float>(size_hidden + 1, size_output - 1));

	typedef nnet::weights<Float,decltype(wspec)> weights;

	using namespace netops;
	auto &&net = softmax_crossentropy(concat_zero(linear_layer<idx::W_hidout>(wspec,
			concat(input_matrix<idx::I_nada>(ispec),
				logistic_sigmoid(linear_layer<idx::W_embhid>(wspec, concat(
					input_matrix<idx::I_nada>(ispec),
					unscaled_dropout<idx::I_mode>(dropout_src, logistic_sigmoid(concat(
						linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_L3>(ispec)),
						linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_L2>(ispec)),
						linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_L1>(ispec)),
						linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_P>(ispec)),
						linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_R1>(ispec)),
						linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_R2>(ispec)),
						linear_layer<idx::W_srcembed>(wspec, input_matrix<idx::I_R3>(ispec))))),
					sample(nn6_combiner<idx::I_antmap>(
						linear_layer<idx::W_antembed>(wspec, input_matrix<idx::I_A>(ispec)),
						logistic_sigmoid(linear_layer<idx::W_V>(wspec,
							logistic_sigmoid(linear_layer<idx::W_U>(wspec,
								input_matrix<idx::I_T>(ispec))))))))))))));

	typedef typename std::remove_reference<decltype(net)>::type::expr_type net_type;
	return nn6<Float,decltype(wspec),net_type>(std::move(wspec), std::move(net));
}

template<class Float,class Net,class Spec>
template<class InputType>
auto nn6<Float,Net,Spec>::operator()(const weight_type &weights, const InputType &input) {
	return net_.fprop(input, weights.sequence());
}

template<class Float,class Net,class Spec>
template<class InputType,class OutputType>
void nn6<Float,Net,Spec>::bprop(const InputType &input, const OutputType &targets, const weight_type &weights, weight_type &grads) {
	net_.bprop_loss(targets, input, weights.sequence(), grads.sequence());
}

template<class InputSeq,class Targets>
auto nn6_dataset<InputSeq,Targets>::subset(std::size_t from, std::size_t to) const {
	const auto &A = netops::at_spec<idx::I_A>(input_);
	const auto &T = netops::at_spec<idx::I_T>(input_);
	const auto &antmap = netops::at_spec<idx::I_antmap>(input_);
	const auto &nada = netops::at_spec<idx::I_nada>(input_);
	const auto &srcctx = netops::at_spec<idx::I_srcctx>(input_);
	const auto &mode = netops::at_spec<idx::I_mode>(input_);
	int nexmpl = std::min(to, nitems()) - from;
	int from_ant = antmap.head(from).sum();
	int n_ant = antmap.middleRows(from, nexmpl).sum();
	return make_nn6_dataset(nlink_,
		fusion::make_vector(
			fusion::make_vector(A.middleRows(from_ant, n_ant).eval(), T.middleRows(from_ant, n_ant).eval(),
				antmap.middleRows(from, nexmpl).eval(), nada.middleRows(from, nexmpl).eval()),
			fusion::as_vector(fusion::transform(srcctx, [from, nexmpl] (const auto &m) { return m.middleRows(from, nexmpl).eval(); })),
			mode),
		targets_.middleRows(from, nexmpl).eval());
}

} // namespace nn6

#endif

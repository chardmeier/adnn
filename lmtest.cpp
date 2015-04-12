#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/range_c.hpp>

#include "logbilinear_lm.h"
#include "nnopt.h"

template<int Order,class FF>
typename nnet::lblm<Order,FF>::dataset lblm_load_data(const char *file, const typename nnet::lblm<Order,FF>::dataset::vocmap_type &vocmap);
template<int Order,class FF>
typename nnet::lblm<Order,FF>::dataset lblm_load_data(const char *file, const typename nnet::lblm<Order,FF>::dataset::vocmap_type *vocmap = NULL);

template<int Order,class FF>
typename nnet::lblm<Order,FF>::dataset lblm_load_data(const char *file, const typename nnet::lblm<Order,FF>::dataset::vocmap_type &vocmap) {
	return lblm_load_data<Order,FF>(file, &vocmap);
}

namespace {
	template<int Order,class OutputType,class Idx>
	struct process_ngram {
		typedef typename std::remove_reference<OutputType>::type output_type;

		output_type &out_;
		std::size_t row_;
		Idx *p_;

		process_ngram(output_type &out, std::size_t row, Idx *p) : out_(out), row_(row), p_(p) {}

		template<class T>
		void operator()(T i) {
			typedef typename output_type::float_type FF;
			out_.template at<Order - T::value - 1>().insert(row_, *p_) = 1;
			if(*p_ != 0)
				p_--;
		}
	};
}

template<int Order,class FF>
typename nnet::lblm<Order,FF>::dataset lblm_load_data(const char *file, const typename nnet::lblm<Order,FF>::dataset::vocmap_type *vocmap) {
	typedef typename nnet::lblm<Order,FF>::dataset::vocidx_type idx;
	std::vector<idx> corpus;
	const idx SENTENCE_BOUNDARY = 0;
	const idx UNKNOWN_WORD = 1;
	idx vocsize = 2;

	typename nnet::lblm<Order,FF>::dataset out;
	typedef typename nnet::lblm<Order,FF>::dataset::vocmap_type vocmap_type;
 
	if(vocmap != NULL) {
		out.vocmap() = *vocmap;
		vocsize = out.vocmap().size();
	} else {
		out.vocmap().insert(std::make_pair("</s>", SENTENCE_BOUNDARY));
		out.vocmap().insert(std::make_pair("<unk>", UNKNOWN_WORD));
	}
	
	std::size_t nwords = 0;
	std::ifstream is(file);
	corpus.push_back(SENTENCE_BOUNDARY);
	for(std::string line; getline(is, line);) {
		std::istringstream ts(line);
		for(std::string token; getline(ts, token, ' ');) {
			typename vocmap_type::iterator it = out.vocmap().find(token);
			idx tokidx;
			if(it == out.vocmap().end()) {
				if(vocmap != NULL)
					tokidx = UNKNOWN_WORD;
				else {
					tokidx = vocsize++;
					out.vocmap().insert(std::make_pair(token, tokidx));
				}
			}
			corpus.push_back(tokidx);
			nwords++;
		}
		corpus.push_back(SENTENCE_BOUNDARY);
	}

	auto setup_matrix = [&] (nnet::sparse_matrix<FF> &mat) {
			mat.resize(corpus.size() - 1, vocsize);
			mat.reserve(Eigen::VectorXi::Constant(corpus.size() - 1, 1));
		};
	boost::fusion::for_each(out.inputs().sequence(), setup_matrix);
	boost::fusion::for_each(out.targets().sequence(), setup_matrix);

	for(std::size_t i = 1; i < corpus.size(); i++) { // the first element is just a boundary
		out.targets().template at<0>().insert(i - 1, corpus[i]) = 1;
		boost::mpl::for_each<boost::mpl::range_c<int,0,Order> >
			(process_ngram<Order,decltype(out.inputs()),idx>(out.inputs(), i - 1, &corpus[i-1]));
/*
		for(std::size_t p = i - 1, n = 0; n < Order; n++) {
			out.inputs().template at<Order-n-1>().insert(i, corpus[p], ONE);
			if(corpus[p] != SENTENCE_BOUNDARY)
				p--;
		}
*/
	}

	return out;
}

int main() {
	const int ngram_order = 3;
	typedef nnet::lblm<ngram_order,adept::Real> net_type;

	net_type::dataset trainset = lblm_load_data<ngram_order,adept::Real>("train.txt");
	net_type::dataset valset = lblm_load_data<ngram_order,adept::Real>("val.txt", trainset.vocmap());
	net_type::dataset testset = lblm_load_data<ngram_order,adept::Real>("test.txt", trainset.vocmap());
	
	net_type net(trainset.vocmap().size(), 150);
	nnet::crossentropy_loss loss;

	nnet::nnopt<net_type> opt(net);
	nnet::nnopt_results<net_type> res = opt.train(net, loss, trainset, valset);

	std::cout << "Training energy: ";
	std::copy(res.trainerr.begin(), res.trainerr.end(), std::ostream_iterator<net_type::float_type>(std::cout, " "));
	std::cout << "\nValidation energy: ";
	std::copy(res.valerr.begin(), res.valerr.end(), std::ostream_iterator<net_type::float_type>(std::cout, " "));
	std::cout << std::endl;

	const auto &testout = net(res.best_weights, testset.inputs());
	std::cout << "Test energy: " << evaluate_loss(loss, testout, testset.targets()) << '\n';

	return 0;
}



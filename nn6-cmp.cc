#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "nn6.h"
#include "nnopt.h"

int main(int argc, char **argv) {
	if(argc != 2) {
		std::cerr << "Usage: nn6 input.nn6" << std::endl;
		return 1;
	}

	std::string datafile = argv[1];

/*
	std::size_t size_U = 20;
	std::size_t size_antembed = 20;
	std::size_t size_srcembed = 20;
	std::size_t size_hidden = 50;
*/

	nn6::classmap classmap("/cluster/home/chm/adnn/cmpex/5class.classes", true);
	nn6::tagmap tagmap("/usit/abel/u1/chm/WMT2013.en-fr/pronoun-model/lefff-2.1.utf8.tags");
	nn6::vocmap srcvocmap;
	nn6::vocmap antvocmap;
	auto dataset = nn6::load_nn6<double,true>(datafile, classmap, srcvocmap, antvocmap, tagmap);
	std::cerr << "Data loaded." << std::endl;

	nn6::dump_nn6_dataset("dump", dataset, srcvocmap, antvocmap);

/*
	auto net = nn6::make_nn6<double>(train.input(),
		size_U, size_antembed, size_srcembed, size_hidden, train.nclasses(),
		params.get<double>("nn6.dropout-src", 1.0));
	typedef decltype(net) net_type;

	nnet::nnopt<net_type> opt(net, params.get_child("nn6.nnopt"));
	nnet::nnopt_results<net_type> res = opt.train(net, train, val);

	std::cerr << "Training error: ";
	std::copy(res.trainerr.begin(), res.trainerr.end(), std::ostream_iterator<net_type::float_type>(std::cerr, " "));
	std::cerr << "\nValidation error: ";
	std::copy(res.valerr.begin(), res.valerr.end(), std::ostream_iterator<net_type::float_type>(std::cerr, " "));
	std::cerr << std::endl;

	auto testout = net(res.best_weights, testset.sequence());
	std::cerr << "Test error: " << net.error(testout, testset.targets()) << '\n';
	std::cerr << "Precision/recall:\n" << precision_recall(testout.matrix(), testset.targets().matrix()) << std::endl;

	std::cout << res.best_weights << std::endl;
*/

	return 0;
}



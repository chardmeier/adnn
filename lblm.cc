#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "logbilinear_lm.h"
#include "nnopt.h"

int main(int argc, char **argv) {
	if(argc != 2) {
		std::cerr << "Usage: lblm config.xml" << std::endl;
		return 1;
	}
	boost::property_tree::ptree params;
	boost::property_tree::read_xml(argv[1], params);

	std::string train_lblm = params.get<std::string>("lblm.data.train");
	std::string val_lblm = params.get<std::string>("lblm.data.val");
	std::string test_lblm = params.get<std::string>("lblm.data.test");

	std::size_t size_embed = params.get<std::size_t>("lblm.layers.embed");

	vocmap::vocmap voc;
	auto train = lblm::load_lblm<double,true>(train_lblm, voc);
	auto val = lblm::load_lblm<double,false>(val_lblm, voc);
	auto testset = lblm::load_lblm<double,false>(test_lblm, voc);
	std::cerr << "Data loaded." << std::endl;

	auto net = lblm::make_lblm<double>(train.input(), size_embed);
	typedef decltype(net) net_type;

	nnet::nnopt<net_type> opt(net, params.get_child("lblm.nnopt"));
	nnet::nnopt_results<net_type> res = opt.train(net, train, val);

	std::cerr << "Training error: ";
	std::copy(res.trainerr.begin(), res.trainerr.end(), std::ostream_iterator<net_type::float_type>(std::cerr, " "));
	std::cerr << "\nValidation error: ";
	std::copy(res.valerr.begin(), res.valerr.end(), std::ostream_iterator<net_type::float_type>(std::cerr, " "));
	std::cerr << std::endl;

	auto testout = net(res.best_weights, testset.sequence());
	std::cerr << "Test error: " << net.error(testout, testset.targets()) << '\n';

	std::cout << res.best_weights << std::endl;

	return 0;
}



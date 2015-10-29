#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "nn6.h"
#include "nnopt.h"

template<class Idx,class Seq>
void join_and_dump(const Seq &sequence, const char *outfile) {
	typedef typename boost::mpl::push_back<Idx,boost::mpl::int_<0>>::type WIdx;
	typedef typename boost::mpl::push_back<Idx,boost::mpl::int_<1>>::type BIdx;
	const auto &weights = netops::at_spec<WIdx>(sequence);
	const auto &bias = netops::at_spec<BIdx>(sequence);
	Eigen::MatrixXd concat(weights.rows() + 1, weights.cols());
	concat << bias, weights;
	std::ofstream os(outfile);
	os << concat << '\n';
}

template<class Weights>
void dump_weights(const Weights &ww) {
	join_and_dump<nn6::idx::W_U>(ww.sequence(), "weights.linkAhid");
	join_and_dump<nn6::idx::W_V>(ww.sequence(), "weights.AhidAres");
	join_and_dump<nn6::idx::W_antembed>(ww.sequence(), "weights.antembed");
	join_and_dump<nn6::idx::W_srcembed>(ww.sequence(), "weights.srcembed");
	join_and_dump<nn6::idx::W_embhid>(ww.sequence(), "weights.embhid");
	join_and_dump<nn6::idx::W_hidout>(ww.sequence(), "weights.hidout");
}

int main(int argc, char **argv) {
	if(argc != 2) {
		std::cerr << "Usage: nn6 input.nn6" << std::endl;
		return 1;
	}

	std::string datafile = argv[1];

	std::size_t size_U = 20;
	std::size_t size_antembed = 20;
	std::size_t size_srcembed = 20;
	std::size_t size_hidden = 50;

	nn6::classmap classmap("/cluster/home/chm/adnn/cmpex/5class.classes", true);
	nn6::tagmap tagmap("/usit/abel/u1/chm/WMT2013.en-fr/pronoun-model/lefff-2.1.utf8.tags");
	nn6::vocmap srcvocmap;
	nn6::vocmap antvocmap;
	auto dataset = nn6::load_nn6<double,true>(datafile, classmap, srcvocmap, antvocmap, tagmap);
	std::cerr << "Data loaded." << std::endl;

	nn6::dump_nn6_dataset("dump", dataset, srcvocmap, antvocmap);

	auto net = nn6::make_nn6<double>(dataset.input(),
		size_U, size_antembed, size_srcembed, size_hidden, dataset.nclasses(),
		1.0);
	typedef decltype(net) net_type;

	net_type::weight_type ww(net.spec());
	ww.init_normal(.04);

	dump_weights(ww);

	auto output = net(ww, dataset.sequence());

	std::ofstream output_os("output");
	output_os << output << '\n';

	return 0;
}



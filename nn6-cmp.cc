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
	Eigen::IOFormat dense_format(Eigen::FullPrecision, Eigen::DontAlignCols, " ", "\n", "", "", "", "\n");
	std::ofstream os(outfile);
	os << concat.format(dense_format);
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
	if(argc != 4) {
		std::cerr << "Usage: nn6 train.nn6 val.nn6 test.nn6" << std::endl;
		return 1;
	}

	std::string train_nn6 = argv[1];
	std::string val_nn6 = argv[2];
	std::string test_nn6 = argv[3];

	std::size_t size_U = 20;
	std::size_t size_antembed = 20;
	std::size_t size_srcembed = 20;
	std::size_t size_hidden = 50;

	nn6::classmap classmap("/cluster/home/chm/adnn/cmpex/5class.classes", true);
	nn6::tagmap tagmap("/usit/abel/u1/chm/WMT2013.en-fr/pronoun-model/lefff-2.1.utf8.tags");
	nn6::vocmap srcvocmap;
	nn6::vocmap antvocmap;
	auto train = nn6::load_nn6<double,true>(train_nn6, classmap, srcvocmap, antvocmap, tagmap);
	auto val = nn6::load_nn6<double,false>(val_nn6, classmap, srcvocmap, antvocmap, tagmap, train.nlink());
	auto testset = nn6::load_nn6<double,false>(test_nn6, classmap, srcvocmap, antvocmap, tagmap, train.nlink());
	std::cerr << "Data loaded." << std::endl;

	nn6::dump_nn6_dataset("dump-train", train, srcvocmap, antvocmap);
	nn6::dump_nn6_dataset("dump-val", val, srcvocmap, antvocmap);
	nn6::dump_nn6_dataset("dump-test", testset, srcvocmap, antvocmap);

	auto net = nn6::make_nn6<double>(train.input(),
		size_U, size_antembed, size_srcembed, size_hidden, train.nclasses(),
		1.0);
	typedef decltype(net) net_type;

	net_type::weight_type ww(net.spec());
	ww.init_normal(.04);

	dump_weights(ww);

	auto output = net(ww, train.sequence());

	std::ofstream output_os("output");
	output_os << output << '\n';

	return 0;
}



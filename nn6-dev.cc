#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "nn6-dev.h"
#include "nnopt.h"

template<class Derived1,class Derived2>
auto precision_recall(const Eigen::MatrixBase<Derived1> &pred, const Eigen::MatrixBase<Derived2> &gold);

template<class Derived1,class Derived2>
auto precision_recall(const Eigen::MatrixBase<Derived1> &pred, const Eigen::MatrixBase<Derived2> &gold) {
	typedef typename Derived1::Scalar F;
	const auto &hardpred = (pred.cwiseEqual(pred.rowwise().maxCoeff().replicate(1, pred.cols()))).template cast<F>();
	const auto &match = (hardpred.array() * gold.array()).colwise().sum();
	Eigen::Array<F,3,Eigen::Dynamic> pr(3, pred.cols());
	pr.row(0) = match / hardpred.array().colwise().sum();
	pr.row(1) = match / gold.array().colwise().sum();
	pr.row(2) = 2 * pr.row(0) * pr.row(1) / (pr.row(0) + pr.row(1));
	return pr;
}

int main(int argc, char **argv) {
	if(argc != 2) {
		std::cerr << "Usage: nn6 config.xml" << std::endl;
		return 1;
	}
	boost::property_tree::ptree params;
	boost::property_tree::read_xml(argv[1], params);

	std::string train_nn6 = params.get<std::string>("nn6.data.train");
	std::string val_nn6 = params.get<std::string>("nn6.data.val");
	std::string test_nn6 = params.get<std::string>("nn6.data.test");

	std::size_t size_U = params.get<std::size_t>("nn6.layers.U");
	std::size_t size_antembed = params.get<std::size_t>("nn6.layers.antembed");
	std::size_t size_srcembed = params.get<std::size_t>("nn6.layers.srcembed");
	std::size_t size_hidden = params.get<std::size_t>("nn6.layers.hidden");

	nn6_dev::classmap classmap(params.get<std::string>("nn6.classmap"), params.get<bool>("nn6.with-other", true));
	nn6_dev::tagmap tagmap(params.get<std::string>("nn6.tagmap", "/dev/null"));
	nn6_dev::vocmap srcvocmap;
	nn6_dev::vocmap antvocmap;
	auto train = nn6_dev::load_nn6<double,true>(train_nn6, classmap, srcvocmap, antvocmap, tagmap);
	auto val = nn6_dev::load_nn6<double,false>(val_nn6, classmap, srcvocmap, antvocmap, tagmap, train.nlink());
	auto testset = nn6_dev::load_nn6<double,false>(test_nn6, classmap, srcvocmap, antvocmap, tagmap, train.nlink());
	std::cerr << "Data loaded." << std::endl;

	auto net = nn6_dev::make_nn6<double>(train.input(),
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

	return 0;
}



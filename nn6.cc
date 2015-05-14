#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include "nn6.h"
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

int main() {
	nn6::vocmap srcvocmap;
	nn6::vocmap antvocmap;
	//auto data = load_nn6("/work/users/chm/ncv9.nn6", srcvocmap, antvocmap);
	auto data = nn6::load_nn6<double>("small.nn6", srcvocmap, antvocmap);
	const auto &input = data.input();
	const auto &targets = data.targets();
	std::cerr << "Data loaded." << std::endl;

	auto net = nn6::make_nn6<double>(input, targets, 20, 20, 20, 50);
	typedef decltype(net) net_type;

	nnet::nnopt<net_type> opt(net);
	nnet::nnopt_results<net_type> res = opt.train(net, data, data);

	std::cout << "Training error: ";
	std::copy(res.trainerr.begin(), res.trainerr.end(), std::ostream_iterator<net_type::float_type>(std::cout, " "));
	std::cout << "\nValidation error: ";
	std::copy(res.valerr.begin(), res.valerr.end(), std::ostream_iterator<net_type::float_type>(std::cout, " "));
	std::cout << std::endl;

/*
	const auto &testout = net(res.best_weights, testset.inputs());
	std::cout << "Test error: " << evaluate_loss(loss, testout, testset.targets()) << '\n';
	std::cout << "Precision/recall:\n" << precision_recall(testout.matrix(), testset.targets().matrix()) << std::endl;
*/

	return 0;
}



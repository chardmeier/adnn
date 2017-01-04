#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include "mlp.h"
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
	mlp::mlp_dataset<double> data;

	#include "seeds.data"

	auto net = mlp::make_mlp<double>(data.sequence(), 100);
	typedef decltype(net) net_type;

	std::size_t split1 = std::floor(.6 * data.nitems());
	std::size_t split2 = std::floor(.8 * data.nitems());

	data.shuffle();
	const auto &trainset = data.subset(0, split1);
	const auto &valset = data.subset(split1, split2);
	const auto &testset = data.subset(split2, data.nitems());
	
	nnet::nnopt<net_type> opt(net);
	nnet::nnopt_results<net_type> res = opt.train(net, trainset, valset);

	std::cout << "Training error: ";
	std::copy(res.trainerr.begin(), res.trainerr.end(), std::ostream_iterator<net_type::float_type>(std::cout, " "));
	std::cout << "\nValidation error: ";
	std::copy(res.valerr.begin(), res.valerr.end(), std::ostream_iterator<net_type::float_type>(std::cout, " "));
	std::cout << std::endl;

/*
	const auto &testout = net(res.best_weights, testset.sequence());
	std::cout << "Test error: " << evaluate_loss(loss, testout, testset.targets()) << '\n';
	std::cout << "Precision/recall:\n" << precision_recall(testout.matrix(), testset.targets().matrix()) << std::endl;
*/

	return 0;
}



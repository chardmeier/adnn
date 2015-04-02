#ifndef NNET_NNOPT_H
#define NNET_NNOPT_H

#include "nnet.h"
#include "net_wrapper.h"

namespace nnet {

template<class Net>
struct nnopt_results {
	typedef typename Net::float_type float_type;
	typedef typename Net::template weight_type<float_type> weight_type;

	nnopt_results(const Net &net) :
		best_weights(net.spec()) {}

	std::vector<float_type> trainerr;
	std::vector<float_type> valerr;
	float_type best_valerr;
	weight_type best_weights;
};

template<class Net>
class nnopt {
public:
	typedef typename Net::float_type float_type;
	typedef typename Net::template weight_type<float_type> weight_type;

private:
	int nsteps_;
	std::size_t batchsize_;
	weight_type init_weights_;
	float_type initial_learning_rate_;
	float_type momentum_;
	float_type l2reg_;

public:
	nnopt(const Net &net);

	template<class Loss,class TrainingDataset,class ValidationDataset>
	nnopt_results<Net> train(const Net &net, const Loss &loss, const TrainingDataset &trainset, const ValidationDataset &valset) const;
};

template<class Net>
nnopt<Net>::nnopt(const Net &net) :
		nsteps_(10), batchsize_(50), init_weights_(net.spec()), initial_learning_rate_(.001), momentum_(.9), l2reg_(.001) {
	init_weights_.init_normal(.01);
}

template<class Net>
template<class Loss,class TrainingDataset,class ValidationDataset>
nnopt_results<Net> nnopt<Net>::train(const Net &net, const Loss &loss, const TrainingDataset &trainset, const ValidationDataset &valset) const {
	const float_type ONE = 1;
	const float_type ZERO = 0;
	const float_type TINY = 1e-15;

	nnopt_results<Net> results(net);

	net_wrapper<Net,Loss> wrapped_net(net, loss);

	weight_type ww(init_weights_);

	weight_type gain(net.spec(), ONE);
	weight_type weight_change(net.spec(), ZERO);
	weight_type rms(net.spec(), ONE);
	weight_type prev_grad(net.spec(), ZERO);

	float_type alpha = initial_learning_rate_;

	results.best_valerr = std::numeric_limits<float_type>::infinity();
	
	bool first_iteration = true;
	for(int i = 0; i < nsteps_; i++) {
		float_type err = 0;
		for(auto batchit = trainset.batch_begin(batchsize_); batchit != trainset.batch_end(); ++batchit) {
			weight_type grad(net.spec(), ZERO);
			err += wrapped_net(ww, batchit->inputs(), batchit->targets(), grad);
			grad.array() += l2reg_ * ww.array();
			//std::cerr << "grad.w1:\n" << grad.w1() << std::endl;
			rms.array() = float_type(.9) * rms.array() + float_type(.1) * grad.array() * grad.array();
			//std::cerr << "rms.w1:\n" << rms.w1() << std::endl;
			weight_type normgrad(net.spec());
			normgrad.array() = grad.array() / (rms.array().sqrt() + TINY);
			//std::cerr << "normgrad.w1:\n" << normgrad.w1() << std::endl;
			weight_change.array() = momentum_ * weight_change.array() - alpha * gain.array() * normgrad.array();
			ww.array() += weight_change.array();
			//std::cerr << "weight_change.w1:\n" << weight_change.w1() << std::endl;
			//std::cerr << "- ww.w1:\n" << ww.w1() << std::endl;

			if(!first_iteration) {
				std::reference_wrapper<bool(float_type)> sign(std::signbit);
				gain.array() = 
					(grad.array().unaryExpr(sign) == prev_grad.array().unaryExpr(sign)).
					select(gain.array() + float_type(.05), float_type(.95) * gain.array());
			} else
				first_iteration = false;

			prev_grad = grad;
		}
		results.trainerr.push_back(err);

		const auto &valout = net(ww, valset.inputs());
		//std::cerr << "ww.w1\n" << ww.w1() << "\nvalout:\n" << valout.matrix() << std::endl;
		results.valerr.push_back(evaluate_loss(loss, valout, valset.targets())); 
		if(results.valerr.back() < results.best_valerr) {
			results.best_valerr = results.valerr.back();
			results.best_weights = ww;
		}

		std::cout << i << " (" << alpha << "): Training error: " << results.trainerr.back() <<
			", validation error: " << results.valerr.back() << std::endl;
	}

	return results;
}

} // namespace nnet

#endif

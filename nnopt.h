#ifndef NNET_NNOPT_H
#define NNET_NNOPT_H

#include "nnet.h"

#include <chrono>
#include <ctime>
#include <random>

namespace nnet {

template<class Net>
struct nnopt_results {
	typedef typename Net::float_type float_type;
	typedef typename Net::weight_type weight_type;

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
	typedef typename Net::weight_type weight_type;

private:
	int nsteps_;
	std::size_t batchsize_;
	weight_type init_weights_;
	float_type initial_learning_rate_;
	float_type rate_decay_;
	float_type learning_schedule_;
	bool rate_heuristic_;
	float_type momentum_;
	float_type l2reg_;

	bool output_detailed_timing_;

public:
	nnopt(const Net &net);

	template<class Params>
	nnopt(const Net &net, const Params &params);

	template<class TrainingDataset,class ValidationDataset>
	nnopt_results<Net> train(Net &net, const TrainingDataset &trainset, const ValidationDataset &valset) const;
};

template<class Net>
nnopt<Net>::nnopt(const Net &net) :
		nsteps_(1), batchsize_(100), init_weights_(net.spec()),
		initial_learning_rate_(.001), rate_decay_(1), learning_schedule_(20),
		rate_heuristic_(false), momentum_(.9), l2reg_(.001),
		output_detailed_timing_(false) {
	init_weights_.init_normal(.01);
}

template<class Net>
template<class Params>
nnopt<Net>::nnopt(const Net &net, const Params &params) :
		nsteps_(params.template get<int>("nsteps", 1)),
		batchsize_(params.template get<std::size_t>("batchsize", 100)),
		init_weights_(net.spec()),
		initial_learning_rate_(params.template get<float_type>("learning-rate", .001)),
		rate_decay_(params.template get<float_type>("rate-decay", 1)),
		learning_schedule_(params.template get<float_type>("learning-schedule", .001)),
		rate_heuristic_(params.template get<bool>("rate-heuristic", false)),
		momentum_(params.template get<float_type>("momentum", .001)),
		l2reg_(params.template get<float_type>("l2reg", .001)),
		output_detailed_timing_(params.template get<bool>("output-detailed-timing", false)) {
	init_weights_.init_normal(params.template get<float_type>("stddev", .01));
}

template<class Net>
template<class TrainingDataset,class ValidationDataset>
nnopt_results<Net> nnopt<Net>::train(Net &net, const TrainingDataset &trainset, const ValidationDataset &valset) const {
	const float_type ONE = 1;
	const float_type ZERO = 0;
	const float_type TINY = 1e-15;

	std::random_device rnddev;
	std::default_random_engine rndeng(rnddev());
	std::uniform_real_distribution<float_type> flip_coin(0, 1);

	nnopt_results<Net> results(net);

	weight_type ww(init_weights_);

	weight_type gain(net.spec(), ONE);
	weight_type weight_change(net.spec(), ZERO);
	weight_type rms(net.spec(), ONE);
	weight_type prev_grad(net.spec(), ZERO);

	results.best_valerr = std::numeric_limits<float_type>::infinity();
	
	float_type nbatches = std::ceil(float_type(trainset.nitems()) / batchsize_);
	std::size_t progress = nbatches / 80 + 1;

	bool first_iteration = true;
	float_type alpha = initial_learning_rate_;
	int alphachange_steps = 0;
	for(int i = 0; i < nsteps_; i++) {
		float_type err = 0;
		if(learning_schedule_ > 0)
			alpha = alpha / (ONE + float_type(i) / learning_schedule_);
		std::size_t batchcnt = 0;
		std::chrono::system_clock::time_point t_epoch_start = std::chrono::system_clock::now();
		for(auto batchit = trainset.batch_begin(batchsize_); batchit != trainset.batch_end(); ++batchit, ++batchcnt) {
			weight_type grad(net.spec(), ZERO);
			std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
			auto output = net(ww, batchit->sequence());
			std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
			net.bprop(batchit->sequence(), batchit->targets(), ww, grad);
			err += net.error(output, batchit->targets()) / nbatches;
			std::chrono::system_clock::time_point t3 = std::chrono::system_clock::now();
			grad.array() += l2reg_ * ww.array();
			rms.array() = float_type(.9) * rms.array() + float_type(.1) * grad.array() * grad.array();
			weight_type normgrad(net.spec());
			normgrad.array() = grad.array() / (rms.array().sqrt() + TINY);
			weight_change.array() = momentum_ * weight_change.array() - alpha * gain.array() * normgrad.array();
			ww.array() += weight_change.array();

			if(!first_iteration) {
				struct {
					bool operator()(const float_type &x) const {
						return std::signbit(x);
					}
				} sign;
				gain.array() = 
					(grad.array().unaryExpr(sign) == prev_grad.array().unaryExpr(sign)).
					select(gain.array() + float_type(.05), float_type(.95) * gain.array());
			} else
				first_iteration = false;

			prev_grad = grad;

			std::chrono::system_clock::time_point t4 = std::chrono::system_clock::now();
			if(output_detailed_timing_)
				std::cerr <<
					"fprop: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us - " <<
					"bprop: " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << "us - " <<
					"sgd: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << "us - " <<
					"err: " << err << std::endl;

			if(batchcnt % progress == 0 && batchcnt > 0)
				//std::cerr << "batchcnt: " << batchcnt << std::endl; // '.';
				std::cerr << '.';
		}
		results.trainerr.push_back(err);

		auto valout = net(ww, valset.sequence());
		//std::cerr << "ww.w1\n" << ww.w1() << "\nvalout:\n" << valout.matrix() << std::endl;
		results.valerr.push_back(net.error(valout, valset.targets()));
		if(results.valerr.back() < results.best_valerr) {
			results.best_valerr = results.valerr.back();
			results.best_weights = ww;
		}
		std::chrono::system_clock::time_point t_epoch_end = std::chrono::system_clock::now();

		std::time_t t = std::time(nullptr);
		char time[80];
		std::strftime(time, sizeof(time), "%c", std::localtime(&t));
		std::cerr << '\n' << time << " (" <<
			std::chrono::duration_cast<std::chrono::milliseconds>(t_epoch_end - t_epoch_start).count() <<
			"ms): " << i << " (" << alpha << "): Training error: " <<
			results.trainerr.back() << ", validation error: " << results.valerr.back() << std::endl;

		// Learning rate adjustment heuristic that seemed to work well in old Matlab code
		if(rate_heuristic_ && i > 6 && alphachange_steps > 5) {
			int neg = 0;
			for(int j = i - 6; j < i; j++)
				if(results.trainerr[j+1] > results.trainerr[j])
					neg++;
			if(neg > 2) {
				alpha *= float_type(.8);
				std::cerr << "Decreasing learning rate to " << alpha << ".\n";
				alphachange_steps = 0;
			} else {
				float_type prob = float_type(.3) * float_type(6 - neg) / 6;
				if(flip_coin(rndeng) < prob) {
					alpha *= float_type(1.05);
					std::cerr << "Increasing learning rate to " << alpha << ".\n";
					alphachange_steps = 0;
				}
			}
		}

		alphachange_steps++;
		alpha *= rate_decay_;
	}

	return results;
}

} // namespace nnet

#endif

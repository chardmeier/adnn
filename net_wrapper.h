#ifndef NNET_NET_WRAPPER_H
#define NNET_NET_WRAPPER_H

#include <adept.h>

#include "nnet.h"

namespace nnet {

template<class A>
struct ad_types;

template<>
struct ad_types<adept::aReal> {
    typedef adept::Real float_type;
    typedef adept::aReal afloat_type;
};

template<>
struct ad_types<adept::Real> {
    typedef adept::Real float_type;
    typedef adept::aReal afloat_type;
};

template<>
struct value_functor<adept::aReal> {
	typedef typename ad_types<adept::aReal>::float_type result_type;
	result_type operator()(const C &in) const {
		return adept::value(in);
	}
};

struct quadratic_loss {
	template<class T1,class T2>
	struct functor {
		typedef typename T1::float_type result_type;
		result_type operator()(const T1 &output, const T2 &targets) const {
			const auto &err = output.matrix() - targets.matrix();
			return err.cwiseProduct(err).sum() / targets.nitems();
		}
	};
};

struct crossentropy_loss {
	template<class T1,class T2>
	struct functor {
		typedef typename T1::float_type result_type;
		result_type operator()(const T1 &output, const T2 &targets) const {
			return -output.matrix().unaryExpr(log_functor<result_type>()).cwiseProduct(targets.matrix()).sum() / targets.nitems();
		}
	};
};

template<class Loss,class T1,class T2>
typename T1::float_type evaluate_loss(const Loss &loss, const T1 &output, const T2 &targets) {
	typename Loss::template functor<T1,T2> lossfn;
	return lossfn(output, targets);
}

template<class N, class Loss>
class net_wrapper {
public:
	typedef typename N::float_type float_type;
	typedef typename N::template basic_input_type<float_type>::type input_type;
	typedef typename N::template basic_output_type<float_type>::type output_type;
	typedef typename N::template weight_type<float_type> weight_type;

private:
	typedef typename N::afloat_type afloat_type;
	typedef typename N::template basic_input_type<afloat_type>::type ainput_type;
	typedef typename N::template basic_output_type<afloat_type>::type aoutput_type;
	typedef typename N::template weight_type<afloat_type> aweight_type;

	typedef N net_type;

	net_type net_;
	Loss loss_;

public:
	net_wrapper(const net_type &net, const Loss &loss) :
		net_(net), loss_(loss) {}

	template<class InputType,class OutputType>
	float_type operator()(const weight_type &W, const InputType &inp, const OutputType &targets, weight_type &grad) const;
};

template<class N, class Loss>
net_wrapper<N,Loss> wrap_net(const N &net, const Loss &loss) {
	typedef typename N::template output_type<typename N::float_type> otype;
	typedef typename N::template output_type<typename N::afloat_type> aotype;
	return net_wrapper<N,Loss>(net, typename Loss::template functor<aotype>());
}

template<class N, class Loss>
template<class InputType,class OutputType>
typename net_wrapper<N,Loss>::float_type net_wrapper<N,Loss>::operator()(const weight_type &W, const InputType &inp,
		const OutputType &targets, weight_type &grad) const {
	adept::Stack stack;
	aweight_type aW(W);
	ainput_type ainp(inp.template cast<afloat_type>());
	aoutput_type atargets(targets.template cast<afloat_type>());
	stack.new_recording();
	aoutput_type aout = net_(aW, ainp);
	afloat_type err = evaluate_loss(loss_, aout, atargets);
	err.set_gradient(float_type(1));
	stack.compute_adjoint();
	grad = aW.get_gradients();
	//std::cerr << "w1 grad:\n" << grad.w1() << std::endl;
	return err.value();
}

} // namespace nnet

#endif

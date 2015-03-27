#include <iostream>

#include <Eigen/Core>
#include <adept.h>

template<class ActivationFunction,class F = adept::Real,class A = adept::aReal>
class Net;

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

template<class C>
struct value_functor {
	typedef typename ad_types<C>::float_type result_type;
	result_type operator()(const C &in) const {
		return adept::value(in);
	}
};

struct Sigmoid {
	template<class F>
	struct functor {
		typedef F result_type;
		result_type operator()(const F &x) const {
			return F(1) / (F(1) + exp(-x));
		}
	};
};

struct QuadraticLoss {
	template<class T>
	struct functor {
		typedef typename T::float_type result_type;
		result_type operator()(const T &output, const T &targets) const {
			auto err = output - targets;
			return err.cwiseProduct(err).sum();
		}
	};
};

template<class N, class Loss>
class NetWrapper {
public:
	typedef typename N::float_type float_type;
	typedef typename N::template input_type<float_type> input_type;
	typedef typename N::template output_type<float_type> output_type;
	typedef typename N::template weight_type<float_type> weight_type;

private:
	typedef typename N::afloat_type afloat_type;
	typedef typename N::template input_type<afloat_type> ainput_type;
	typedef typename N::template output_type<afloat_type> aoutput_type;
	typedef typename N::template weight_type<afloat_type> aweight_type;

	typedef N net_type;
	typedef typename Loss::template functor<aoutput_type> loss_type;

	net_type net_;
	loss_type loss_;

public:
	NetWrapper(const net_type &net, const loss_type &loss) :
		net_(net), loss_(loss) {}

	template<class OutputType>
	float_type operator()(const weight_type &W, const input_type &inp, const OutputType &targets, weight_type &grad) const;
};

template<class N, class Loss>
NetWrapper<N,Loss> wrap_net(const N &net, const Loss &loss) {
	typedef typename N::template output_type<typename N::float_type> otype;
	typedef typename N::template output_type<typename N::afloat_type> aotype;
	return NetWrapper<N,Loss>(net, typename Loss::template functor<aotype>());
}

template<class N, class Loss>
template<class OutputType>
typename NetWrapper<N,Loss>::float_type NetWrapper<N,Loss>::operator()(const weight_type &W, const input_type &inp,
		const OutputType &targets, weight_type &grad) const {
	adept::Stack stack;
	aweight_type aW(W);
	ainput_type ainp(inp.template cast<afloat_type>());
	aoutput_type atargets(targets.template cast<afloat_type>());
	stack.new_recording();
	aoutput_type aout = net_(aW, ainp);
        std::cerr << "Output: " << aout.value() << '\n';
	afloat_type err = loss_(aout, atargets);
	err.set_gradient(float_type(1));
	stack.compute_adjoint();
	grad = aW.get_gradients();
	return err.value();
}


template<class ActivationFunction,class F,class A>
class Net {
public:
	typedef F float_type;
	typedef A afloat_type;

	template<class FF>
	struct input_type : public Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic> {
		using Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic>::Matrix;
		typedef FF float_type;
	};

	template<class FF>
	struct output_type : public Eigen::Matrix<FF,Eigen::Dynamic,1> {
		using Eigen::Matrix<FF,Eigen::Dynamic,1>::Matrix;
		typedef FF float_type;
	};

	template<class FF>
	struct weight_type {
		typedef FF float_type;

		Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic> w1;
		Eigen::Matrix<FF,Eigen::Dynamic,Eigen::Dynamic> w2;

		weight_type(std::size_t inp, std::size_t hid, std::size_t out) :
				w1(inp, hid), w2(hid, out) {
			w1.setZero();
			w2.setZero();
		}

		template<class FFF>
		weight_type(const weight_type<FFF> &o) :
			w1(o.w1.template cast<FF>()), w2(o.w2.template cast<FF>()) {}

                weight_type<typename ad_types<FF>::float_type> get_gradients() {
                    weight_type<typename ad_types<FF>::float_type> out(w1.rows(), w1.cols(), w2.cols());
                    adept::get_gradients(w1.data(), w1.size(), out.w1.data());
                    adept::get_gradients(w2.data(), w2.size(), out.w2.data());
                    return out;
                }
	};

	template<class FF>
	output_type<FF> operator()(const weight_type<FF> &W, const input_type<FF> &inp) const;
};

template<class ActivationFunction, class F, class A>
template<class FF>
typename Net<ActivationFunction,F,A>::template output_type<FF> Net<ActivationFunction,F,A>::operator()(const weight_type<FF> &w, const input_type<FF> &inp) const {
	typedef typename ActivationFunction::template functor<afloat_type> Activation;

	std::cerr << "w1:\n" << w.w1.unaryExpr(value_functor<FF>()) << '\n';
	std::cerr << "w2:\n" << w.w2.unaryExpr(value_functor<FF>()) << '\n';
	const auto &p1 = inp * w.w1;
	const auto &a1 = p1.unaryExpr(Activation());
	std::cerr << "a1:\n" << a1.unaryExpr(value_functor<FF>()) << '\n';
	const auto &p2 = a1 * w.w2;
	const auto &out = p2.unaryExpr(Activation());
	std::cerr << "out:\n" << out.unaryExpr(value_functor<FF>()) << '\n';
	return out;
}

int main() {
	typedef Net<Sigmoid> net_type;
	typedef NetWrapper<net_type,QuadraticLoss> wrap_type;
	typedef net_type::float_type flt;

	net_type net;
	wrap_type wrap = wrap_net(net, QuadraticLoss());

	const std::size_t inpsize = 4;
	const std::size_t hidsize = 4;
	const std::size_t outsize = 1;

	net_type::input_type<flt> inp(1, inpsize);
	net_type::weight_type<flt> ww(inpsize, hidsize, outsize);
	net_type::weight_type<flt> grad(inpsize, hidsize, outsize);
	net_type::output_type<flt> targets(outsize, 1);

	ww.w1 << 1, 2, 3, 4,
		 5, 6, 7, 8,
		-1,-2,-3,-4,
		-5,-6,-7,-8;

	ww.w2 << 1,2,3,4;

	inp << 1, 2, 3, 4;
	targets << 0;

	flt loss = wrap(ww, inp, targets, grad);

	std::cout << "Loss: " << loss << '\n' <<
		"Gradient:\n" << grad.w1 << "\n---\n" << grad.w2 << std::endl;

	return 0;
}

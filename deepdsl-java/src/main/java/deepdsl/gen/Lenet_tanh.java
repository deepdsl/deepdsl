package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.util.CudaRun;


public class Lenet_tanh extends CudaRun {

	public static void main(String[] args){
		Lenet_tanh run = new Lenet_tanh();
		run.train(1000);
		run.test(10);
		run.save();
		run.free();
	}

	public Lenet_tanh() {
		super("src/main/java/deepdsl/gen/lenet_tanh");
		setTrainData(MnistFactory.getFactory(true, new int[]{500, 1, 28, 28}));
		setTestData(MnistFactory.getFactory(false, new int[]{500, 1, 28, 28}));
	}

	float lrn_rate = -0.1f;

	JCudnnConvolution y8 = addConvolution(new int[]{500,1,28,28},new int[]{20,1,5,5},new int[]{20}, 1, 0);
	JCudnnConvolution y5 = addConvolution(new int[]{500,20,12,12},new int[]{50,20,5,5},new int[]{50}, 1, 0);
	JCudnnPooling y7 = addPooling(new int[]{500,20,24,24}, 2, 2, 0, PoolingType.MAX);
	JCudnnPooling y4 = addPooling(new int[]{500,50,8,8}, 2, 2, 0, PoolingType.MAX);
	JCudnnSoftmax y1 = addSoftmax(new int[]{500,10}, SoftmaxAlgorithm.ACCURATE);
	JCudnnActivation y6 = addActivation(new int[]{500,20,12,12}, ActivationMode.TANH);
	JCudnnActivation y3 = addActivation(new int[]{500,50,4,4}, ActivationMode.TANH);
	JCudnnActivation y2 = addActivation(new int[]{500,500}, ActivationMode.TANH);
	JCudaTensor B = addParam("B", "Constant", 0.0f, 500);
	JCudaTensor B1 = addParam("B1", "Constant", 0.0f, 20);
	JCudaTensor B2 = addParam("B2", "Constant", 0.0f, 50);
	JCudaTensor B3 = addParam("B3", "Constant", 0.0f, 10);
	JCudaTensor Theta = addParam("Theta", "Constant", 0.0f, 10, 500);
	JCudaTensor W = addParam("W", "Random", 0.06793662f, 500, 800);
	JCudaTensor W1 = addParam("W1", "Random", 0.2f, 20, 1, 5, 5);
	JCudaTensor W2 = addParam("W2", "Random", 0.08596024f, 50, 20, 5, 5);

	public float trainFunction(JTensorFloat X, JTensorFloat Y) {
		// val X74 = Cuda(X)
		JCudaTensor X74 = X.asJCudaTensor();
		// val X26 = Convolv(1,0)(X74,W1,B1)
		JCudaTensor X26 = y8.forward(X74, W1, B1);
		// val X27 = Pooling(2,2,0,true)(X26)
		JCudaTensor X27 = y7.forward(X26);
		// val X28 = Tanh()(X27.copy)
		JCudaTensor X28 = y6.forward(X27.clone());
		// val X29 = Convolv(1,0)(X28,W2,B2)
		JCudaTensor X29 = y5.forward(X28, W2, B2);
		// val X30 = Pooling(2,2,0,true)(X29)
		JCudaTensor X30 = y4.forward(X29);
		// val X31 = Tanh()(X30.copy)
		JCudaTensor X31 = y3.forward(X30.clone());
		// val X77 = (X31[1><3])(i10 | @) * (W)(i11 | @)
		JCudaTensor X77 = X31.flatten(1, new int[]{50, 4, 4}).asMatrix(1, true).times(W.asMatrix(1, true));
		// val X33 = (X77 + (i10) => B)
		JCudaTensor X33 = B.copy(500, X77);
		// val X34 = Tanh()(X33)
		JCudaTensor X34 = y2.forward(X33);
		// val X75 = (X34)(i13 | @) * (Theta)(i14 | @)
		JCudaTensor X75 = X34.asMatrix(1, true).times(Theta.asMatrix(1, true));
		// val X35 = (X75 + (i13) => B3)
		JCudaTensor X35 = B3.copy(500, X75);
		// val X36 = Softmax()(X35)
		JCudaTensor X36 = y1.forward(X35);
		// dealloc X35
		X35.free();
		// val X80 = Log X36.copy
		JCudaTensor X80 = X36.clone().log();
		// val X79 = Cuda(Indicator(Y, 10))
		JCudaTensor X79 = Y.asIndicator(10).asJCudaTensor();
		// val X99 = 1/(X36.copy)
		JCudaTensor X99 = X36.clone().pow(-1f);
		// val _loss = ((0 - (X79 . X80)) / |500|)
		float _loss = - X79.dot(X80) / 500f;
		// dealloc X80
		X80.free();
		// val X100 = X79.copy .* X99
		JCudaTensor X100 = X79.clone().times_i(X99);;
		// dealloc X79
		X79.free();
		// dealloc X99
		X99.free();
		// val X101 = - X100
		JCudaTensor X101 = X100.times_i(-1f);;
		// val X37 = (X101 / |500|)
		JCudaTensor X37 = X101.times_i(1 / 500f);;
		// val m1 = (i40) => Theta[@, i40]
		JCudaMatrix m1 = Theta.asMatrix(1, false);
		// val X91 = X37 * d_Softmax()(X36)/d_X35
		JCudaTensor X91 = y1.backward(X37, X36);
		// dealloc X36
		X36.free();
		// dealloc X37
		X37.free();
		// val m4 = (i18) => X91[@, i18]
		JCudaMatrix m4 = X91.asMatrix(1, false);
		// val m5 = (i19) => X34[@, i19]
		JCudaMatrix m5 = X34.asMatrix(1, false);
		// val X88 = (X91)(i39 | @) * m1
		JCudaTensor X88 = X91.asMatrix(1, true).times(m1);
		// B3 = ((Sum(m4) * -0.1) + B3)
		m4.sum(B3, lrn_rate, 1f);
		// val m7 = (i33) => W[@, i33]
		JCudaMatrix m7 = W.asMatrix(1, false);
		// val X83 = X88 * d_Tanh()(X34)/d_X33
		JCudaTensor X83 = y2.backward(X88, X34);
		// Theta = ((m4 * m5 * -0.1) + Theta)
		m4.times(m5, Theta, lrn_rate, 1f);
		// dealloc X34
		X34.free();
		// dealloc X91
		X91.free();
		// val m3 = (i23) => X31[1><3][@, i23]
		JCudaMatrix m3 = X31.flatten(1, new int[]{50, 4, 4}).asMatrix(1, false);
		// val m2 = (i22) => X83[@, i22]
		JCudaMatrix m2 = X83.asMatrix(1, false);
		// val X107 = (X83)(i32 | @) * m7
		JCudaTensor X107 = X83.asMatrix(1, true).times(m7);
		// B = ((Sum(m2) * -0.1) + B)
		m2.sum(B, lrn_rate, 1f);
		// W = ((m2 * m3 * -0.1) + W)
		m2.times(m3, W, lrn_rate, 1f);
		// dealloc X83
		X83.free();
		// val X93 = X107[1<>3] * d_Tanh()(X31)/d_X30
		JCudaTensor X93 = y3.backward(X107.unflatten(1, new int[]{50, 4, 4}), X31);
		// dealloc X31
		X31.free();
		// val X97 = X93 * d_Pooling(2,2,0,true)(X30,X29)/d_X29
		JCudaTensor X97 = y4.backward(X93, X30, X29);
		// dealloc X93
		X93.free();
		// dealloc X30
		X30.free();
		// dealloc X29
		X29.free();
		// B2 = ((X97 * d_Convolv(1,0)()/d_B2 * -0.1) + B2)
		y5.backward_bias(X97, B2, lrn_rate, 1f);
		// val X85 = X97 * d_Convolv(1,0)(W2)/d_X28
		JCudaTensor X85 = y5.backward_data(X97, W2);
		// val X110 = X85 * d_Tanh()(X28)/d_X27
		JCudaTensor X110 = y6.backward(X85, X28);
		// W2 = ((X97 * d_Convolv(1,0)(X28)/d_W2 * -0.1) + W2)
		y5.backward_filter(X97, X28, W2, lrn_rate, 1f);
		// dealloc X97
		X97.free();
		// dealloc X28
		X28.free();
		// val X106 = X110 * d_Pooling(2,2,0,true)(X27,X26)/d_X26
		JCudaTensor X106 = y7.backward(X110, X27, X26);
		// dealloc X110
		X110.free();
		// dealloc X26
		X26.free();
		// dealloc X27
		X27.free();
		// W1 = ((X106 * d_Convolv(1,0)(X74)/d_W1 * -0.1) + W1)
		y8.backward_filter(X106, X74, W1, lrn_rate, 1f);
		// dealloc X74
		X74.free();
		// B1 = ((X106 * d_Convolv(1,0)()/d_B1 * -0.1) + B1)
		y8.backward_bias(X106, B1, lrn_rate, 1f);
		// dealloc X106
		X106.free();

		return _loss; 
	}

	public JCudaTensor testFunction(JTensorFloat X) {
		// val X11 = Cuda(X)
		JCudaTensor X11 = X.asJCudaTensor();
		// val X12 = Convolv(1,0)(X11,W1,B1)
		JCudaTensor X12 = y8.forward(X11, W1, B1);
		// dealloc X11
		X11.free();
		// val X13 = Pooling(2,2,0,true)(X12)
		JCudaTensor X13 = y7.forward(X12);
		// dealloc X12
		X12.free();
		// val X14 = Tanh()(X13)
		JCudaTensor X14 = y6.forward(X13);
		// val X15 = Convolv(1,0)(X14,W2,B2)
		JCudaTensor X15 = y5.forward(X14, W2, B2);
		// dealloc X14
		X14.free();
		// val X16 = Pooling(2,2,0,true)(X15)
		JCudaTensor X16 = y4.forward(X15);
		// dealloc X15
		X15.free();
		// val X17 = Tanh()(X16)
		JCudaTensor X17 = y3.forward(X16);
		// val X22 = (X17[1><3])(i10 | @) * (W)(i11 | @)
		JCudaTensor X22 = X17.flatten(1, new int[]{50, 4, 4}).asMatrix(1, true).times(W.asMatrix(1, true));
		// dealloc X17
		X17.free();
		// val X19 = (X22 + (i10) => B)
		JCudaTensor X19 = B.copy(500, X22);
		// val X20 = Tanh()(X19)
		JCudaTensor X20 = y2.forward(X19);
		// val X24 = (X20)(i13 | @) * (Theta)(i14 | @)
		JCudaTensor X24 = X20.asMatrix(1, true).times(Theta.asMatrix(1, true));
		// dealloc X20
		X20.free();
		// val X21 = (X24 + (i13) => B3)
		JCudaTensor X21 = B3.copy(500, X24);

		return X21; 
	}

}
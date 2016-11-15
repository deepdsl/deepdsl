package deepdsl.cudnn;
 
import static jcuda.jcudnn.JCudnn.cudnnSoftmaxBackward;
import static jcuda.jcudnn.JCudnn.cudnnSoftmaxForward;   
import jcuda.Pointer; 
import deepdsl.cudnn.config.SoftmaxAlgorithm;
import deepdsl.cudnn.config.SoftmaxMode;
import deepdsl.util.ArithStats;

public class JCudnnSoftmax extends JCudaFunction { 
	SoftmaxAlgorithm algorithm; // = SoftmaxAlgorithm.ACCURATE;
	SoftmaxMode mode = SoftmaxMode.CHANNEL; // SoftmaxMode.INSTANCE;
	JCudnnDescriptor dptr; 

	public JCudnnSoftmax(int[] x_dims, SoftmaxAlgorithm algorithm) {
		this.dptr = new JCudnnDescriptor(x_dims);
		this.algorithm = algorithm;
	}
 
	public void free() {
		dptr.free();
	}
	// y = forward(x)
	public JCudaTensor forward(JCudaTensor x) {
		JCudaTensor y = new JCudaTensor(x.getDims());
		return forward(x, y, one, zero);
	}
	// y = forward(x) * alpha + beta * y
	public JCudaTensor forward(JCudaTensor x, JCudaTensor y, Pointer alpha, Pointer beta) {
		long begin = System.nanoTime();
		
		int ret = cudnnSoftmaxForward(cudnnHandle, algorithm.value(), mode.value(), alpha,
				dptr.descriptor, x.getData(), beta, dptr.descriptor,
				y.getData()); 
		 
		checkError(ret);
		
		ArithStats.cuda_timing("softmax forward", begin);
		return y;
	}
	// dx = backward(dy, y)
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y) {
		JCudaTensor dx = new JCudaTensor(y.getDims());
		return backward(dy, y, dx, one, zero);
	}
	// dx = backward(dy, y) * alpha + dx * beta
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y, JCudaTensor dx, Pointer alpha, Pointer beta) { 
		long begin = System.nanoTime();
 
		int ret = cudnnSoftmaxBackward(cudnnHandle, algorithm.value(),
				mode.value(), alpha, dptr.descriptor,
				y.getData(), dptr.descriptor, dy.getData(),
				beta, dptr.descriptor, dx.getData());
		 
		checkError(ret);
		
		ArithStats.cuda_timing("softmax backward", begin);
		return dx;
	}
 
}

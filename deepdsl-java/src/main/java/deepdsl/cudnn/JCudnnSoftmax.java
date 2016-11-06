package deepdsl.cudnn;
 
import static jcuda.jcudnn.JCudnn.cudnnSoftmaxBackward;
import static jcuda.jcudnn.JCudnn.cudnnSoftmaxForward;   
import jcuda.Pointer;
import jcuda.vec.VecFloat;
import deepdsl.cudnn.config.softmax.SoftmaxAlgorithm;
import deepdsl.cudnn.config.softmax.SoftmaxMode;
import deepdsl.util.ArithStats;

public class JCudnnSoftmax extends JCudaFunction { 
	int algorithm; // = SoftmaxAlgorithm.ACCURATE.algorithm();
	int mode = SoftmaxMode.CHANNEL.mode(); // SoftmaxMode.INSTANCE.mode();
	JCudnnDescriptor dptr; 

	public JCudnnSoftmax(int[] x_dims, int algorithm) {
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
		
		int ret = cudnnSoftmaxForward(cudnnHandle, algorithm, mode, alpha,
				dptr.descriptor, x.getData(), beta, dptr.descriptor,
				y.getData()); 
		 
		checkError(ret);
		
		// uniformly add the smallest float to the result of softmax (if it is not log softmax) to prevent underflow 
		if(algorithm != SoftmaxAlgorithm.LOG.algorithm()) {
			float smallest = Float.MIN_NORMAL;
			VecFloat.addScalar(y.size, y.getData(), y.getData(), smallest);
		}
		
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
 
		int ret = cudnnSoftmaxBackward(cudnnHandle, algorithm,
				mode, alpha, dptr.descriptor,
				y.getData(), dptr.descriptor, dy.getData(),
				beta, dptr.descriptor, dx.getData());
		 
		checkError(ret);
		
		ArithStats.cuda_timing("softmax backward", begin);
		return dx;
	}
 
}

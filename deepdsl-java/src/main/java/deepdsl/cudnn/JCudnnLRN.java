package deepdsl.cudnn;
 
import deepdsl.util.ArithStats;
import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnLRNDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

public class JCudnnLRN extends JCudaFunction {
	cudnnLRNDescriptor normDesc = new cudnnLRNDescriptor();
	JCudnnDescriptor dptr;  
	int mode = 0; // LRN mode = 0 ? 
	
	public JCudnnLRN(int[] x_dims, int lrnN, double lrnAlpha, double lrnBeta) {
		this(x_dims, lrnN, lrnAlpha, lrnBeta, 2);
	}
	
	public JCudnnLRN(int[] x_dims, int lrnN, double lrnAlpha, double lrnBeta, int lrnK) {
		this.dptr = new JCudnnDescriptor(x_dims);
		checkError(JCudnn.cudnnCreateLRNDescriptor(normDesc));
		checkError(JCudnn.cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK));
	}
	
	public void free() {
		dptr.free();
		checkError(JCudnn.cudnnDestroyLRNDescriptor(normDesc));
	}
	// y = forward(x)
	public JCudaTensor forward(JCudaTensor x) {
		JCudaTensor y = new JCudaTensor(x.getDims()); 
		return forward(x, y);
	}
	// y = forward(x), where y is passed in
	public JCudaTensor forward(JCudaTensor x, JCudaTensor y) {
		return forward(x, y, one, zero);
	}
	// y = forward(x) * alpha + y * beta
	public JCudaTensor forward(JCudaTensor x, JCudaTensor y, Pointer alpha, Pointer beta) {
		long begin = System.nanoTime();
		
		cudnnTensorDescriptor dptr = this.dptr.descriptor;
		 
		int ret = JCudnn.cudnnLRNCrossChannelForward(cudnnHandle, normDesc, mode,
				alpha, dptr, x.getData(), beta, dptr, y.getData()); 
		 
		checkError(ret);
		
		ArithStats.cuda_timing("LRN forward", begin);
		return y;
	}
	// dy = backward(dy, y, x)
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y, JCudaTensor x) {
		JCudaTensor dx = dy; // can be in place
		return backward(dy, y, x, dx);
	}
	// dx = backward(dy, y, x)
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y, JCudaTensor x, JCudaTensor dx) {
		return backward(dy, y, x, dx, one, zero);
	}
	// dx = backward(dy, y, x) * alpha + dx * beta
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y, JCudaTensor x, JCudaTensor dx, Pointer alpha, Pointer beta) {
		long begin = System.nanoTime();
		cudnnTensorDescriptor dptr = this.dptr.descriptor;
		
		int ret = JCudnn.cudnnLRNCrossChannelBackward(cudnnHandle, normDesc, mode, 
				one, dptr, y.getData(), dptr, dy.getData(),
				dptr, x.getData(), zero, dptr, dx.getData());
		 
		checkError(ret);
		
		ArithStats.cuda_timing("LRN backward", begin);
		return dx;
	} 
}

package deepdsl.cudnn;

import static jcuda.jcudnn.JCudnn.cudnnActivationBackward;
import static jcuda.jcudnn.JCudnn.cudnnActivationForward; 
import deepdsl.cudnn.config.ActivationMode;
import deepdsl.util.ArithStats;
import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor; 

public class JCudnnActivation extends JCudaFunction {
	cudnnActivationDescriptor activationDesc = new cudnnActivationDescriptor();
	JCudnnDescriptor dptr;  
	
	public JCudnnActivation(int[] x_dims, ActivationMode mode) {
		this.dptr = new JCudnnDescriptor(x_dims);
		checkError(JCudnn.cudnnCreateActivationDescriptor(activationDesc));			 // 2nd argument: see ActivationMode class
		checkError(JCudnn.cudnnSetActivationDescriptor(activationDesc, mode.value(), 0, 7)); // 3rd argument: 0 means doesn not propagate NaN
																		 			 // 4th argument: the ceiling for clipped relu
																		   			 //    7 works without gradient clipping
	}
	 
	public void free() {
		dptr.free();
		JCudnn.cudnnDestroyActivationDescriptor(activationDesc);
	}
	
	// x = forward(x)
	public JCudaTensor forward(JCudaTensor x) {
		JCudaTensor y = x; // new JCudaTensor(x.getDims()); // It can be in-place.
		return forward(x, y);
	}
	// y = forward(x)
	public JCudaTensor forward(JCudaTensor x, JCudaTensor y) {
		return forward(x, y, one, zero);
	}
	// y = forward(x) * alpha + y * beta
	public JCudaTensor forward(JCudaTensor x, JCudaTensor y, Pointer alpha, Pointer beta) {
		long begin = System.nanoTime();
		
		cudnnTensorDescriptor dptr = this.dptr.descriptor;
		 
		int ret = cudnnActivationForward(cudnnHandle, activationDesc,
				alpha, dptr, x.getData(), beta, dptr, y.getData()); 
		 
		checkError(ret);
		
		ArithStats.cuda_timing("activation forward", begin);
		return y;
	}
	
	// dy = backward(dy, y, y)
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y) {
		JCudaTensor dx = dy; // new JCudaTensor(x.getDims()); // It can be in-place
		return backward(dy, y, dx);
	}
	
	// dx = backward(dy, y, y)
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y, JCudaTensor dx) {
		return backward(dy, y, dx, one, zero);
	}
 
	// dx = backward(dy, y, y) * alpha + dx * beta
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y, JCudaTensor dx, Pointer alpha, Pointer beta) {
		// x is the same as y. Who knows why CUDNN wants both x and y while only using y
		return backward(dy, y, y, dx, alpha, beta);
	}
	
	// dx = backward(dy, y, x) * alpha + dx * beta 
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y, JCudaTensor x, JCudaTensor dx, Pointer alpha, Pointer beta) {
		long begin = System.nanoTime();

		cudnnTensorDescriptor dptr = this.dptr.descriptor;
		
		int ret = cudnnActivationBackward(cudnnHandle, activationDesc, alpha,
				dptr, y.getData(), dptr, dy.getData(),
				dptr, x.getData(), beta, dptr, dx.getData());
		 
		checkError(ret);
		
		ArithStats.cuda_timing("activation backward", begin);
		return dx;
	} 
}

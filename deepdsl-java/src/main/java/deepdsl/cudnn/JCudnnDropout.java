package deepdsl.cudnn;

import deepdsl.util.ArithStats;
import jcuda.Pointer;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnDropoutDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor; 

public class JCudnnDropout extends JCudaFunction {
	cudnnDropoutDescriptor dropoutDesc = new cudnnDropoutDescriptor();
	JCudnnDescriptor dptr;  
	long seed = 0;
	Pointer reserve = new Pointer();
	long[] reserveSize = {0};
	Pointer states = new Pointer();
	
	public JCudnnDropout(int[] x_dims, float dropout) {
		this.dptr = new JCudnnDescriptor(x_dims);
		JCudnn.cudnnCreateDropoutDescriptor(dropoutDesc);
		long[] stateSize = {0};
		checkError(JCudnn.cudnnDropoutGetStatesSize(cudnnHandle, stateSize));
		checkError(JCudnn.cudnnDropoutGetReserveSpaceSize(dptr.descriptor, reserveSize));
		allocByte(states, stateSize[0]);
		checkError(JCudnn.cudnnSetDropoutDescriptor(dropoutDesc, cudnnHandle, dropout, states, stateSize[0], seed));
		allocByte(reserve, reserveSize[0]);
	}
	
	public void free() {
		dptr.free();
		checkError(JCudnn.cudnnDestroyDropoutDescriptor(dropoutDesc));
		free(states);
		free(reserve);
	}
	// y = forward(x)
	public JCudaTensor forward(JCudaTensor x) {
		long begin = System.nanoTime();
		JCudaTensor y = new JCudaTensor(x.getDims()); 
		
		cudnnTensorDescriptor dptr = this.dptr.descriptor;
		 
		int ret = JCudnn.cudnnDropoutForward(cudnnHandle, dropoutDesc,  
				dptr, x.getData(), dptr, y.getData(),
				reserve, reserveSize[0]); 
		 
		checkError(ret);
		
		ArithStats.cuda_timing("Dropout forward", begin);
		return y;
	}
	// dx = backward(dy)
	public JCudaTensor backward(JCudaTensor dy) {
		long begin = System.nanoTime();
		JCudaTensor dx = new JCudaTensor(dy.getDims());

		cudnnTensorDescriptor dptr = this.dptr.descriptor;
		
		int ret = JCudnn.cudnnDropoutBackward(cudnnHandle, dropoutDesc,  
				dptr, dy.getData(), dptr, dx.getData(), 
				reserve, reserveSize[0]);
		 
		checkError(ret);
		
		ArithStats.cuda_timing("Dropout backward", begin);
		return dx;
	} 
}

package deepdsl.cudnn;

import static jcuda.jcudnn.JCudnn.cudnnCreatePoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnDestroyPoolingDescriptor;
import static jcuda.jcudnn.JCudnn.cudnnGetPoolingNdForwardOutputDim;
import static jcuda.jcudnn.JCudnn.cudnnPoolingBackward;
import static jcuda.jcudnn.JCudnn.cudnnPoolingForward;
import static jcuda.jcudnn.JCudnn.cudnnSetPoolingNdDescriptor; 
import jcuda.Pointer;
import jcuda.jcudnn.cudnnPoolingDescriptor;  
import deepdsl.cudnn.config.PoolingType;
import deepdsl.util.ArithStats;

public class JCudnnPooling extends JCudaFunction {
	private cudnnPoolingDescriptor poolingDesc;
	private int poolDimSize = 2; // number of pooling dimensions
	private int[] window_array = {2, 2}; // pooling window
	private int[] padding_array = {0, 0}; 
	private int[] stride_array = {2, 2}; 
	private int nanOption = 0; // do not propagate NaN
	JCudnnDescriptor x_dptr, y_dptr;
	int[] y_dims;

	public JCudnnPooling(int[] x_dims) {
		this(x_dims, 2);
	}
	public JCudnnPooling(int[] x_dims, int window) {
		this(x_dims, window, window, 0);
	}
	public JCudnnPooling(int[] x_dims, int window, int stride, int padding) {
		this(x_dims, window, stride, padding, PoolingType.MAX); // default to max pooling
	}
	public JCudnnPooling(int[] x_dims, int window, int stride, int padding, PoolingType poolType) { 
		this.window_array = new int[] {window, window};
		this.stride_array = new int[] {stride, stride};
		this.padding_array = new int[] {padding, padding}; // (x + window + padding - stride) % stride == 0
		
		this.poolingDesc = new cudnnPoolingDescriptor();
		checkError(cudnnCreatePoolingDescriptor(poolingDesc));
		
		checkError(cudnnSetPoolingNdDescriptor(poolingDesc,
				poolType.value(), nanOption, poolDimSize,
				window_array, padding_array,
				stride_array));
		x_dptr = new JCudnnDescriptor(x_dims);
		y_dims = new int[x_dims.length];
		
		checkError(cudnnGetPoolingNdForwardOutputDim(poolingDesc, x_dptr.descriptor,
				y_dims.length, y_dims));
		
		y_dptr = new JCudnnDescriptor(y_dims);
	}
	
	public void free() {
		cudnnDestroyPoolingDescriptor(poolingDesc); 
		x_dptr.free();
		y_dptr.free();
	}
	// y = forward(x)
	public JCudaTensor forward(JCudaTensor x) {
		JCudaTensor y = new JCudaTensor(y_dims);
		return forward(x, y, one, zero);
	}
	// y = forward(x) * alpha + y * beta
	public JCudaTensor forward(JCudaTensor x, JCudaTensor y, Pointer alpha, Pointer beta) {
		long begin = System.nanoTime();
		
		int ret = cudnnPoolingForward(cudnnHandle, poolingDesc, alpha, x_dptr.descriptor,
				x.getData(), beta, y_dptr.descriptor, y.getData());
		 
		checkError(ret);
		
		ArithStats.cuda_timing("pooling forward", begin);
		return y;
	}
	// dx = backward(dy, y, x)
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y, JCudaTensor x) {
		JCudaTensor dx = new JCudaTensor(x.getDims());
		return backward(dy, y, x, dx, one, zero); 
	}
	// dx += backward(dy, y, x)
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y, JCudaTensor x, JCudaTensor dx) {
		return backward(dy, y, x, dx, one, one);
	}
	// dx = backward(dy, y, x) * alpha + dx * beta
	public JCudaTensor backward(JCudaTensor dy, JCudaTensor y, JCudaTensor x, JCudaTensor dx, Pointer alpha, Pointer beta) {
		long begin = System.nanoTime(); 
		
		int ret = cudnnPoolingBackward(cudnnHandle, poolingDesc, alpha, y_dptr.descriptor,
				y.getData(), y_dptr.descriptor, dy.getData(),
				x_dptr.descriptor, x.getData(), beta, x_dptr.descriptor, dx.getData());
 
		checkError(ret);
		
		ArithStats.cuda_timing("pooling backward", begin);
		return dx;
	}
	
}

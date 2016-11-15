package deepdsl.cudnn;
 
 
import java.util.Arrays;
import static deepdsl.cudnn.JCudaFunction.checkError;

import deepdsl.cudnn.config.TensorDataType;
import deepdsl.cudnn.config.TensorFormat;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnTensorDescriptor;

public class JCudnnDescriptor {
	static TensorDataType dataType = TensorDataType.FLOAT;
	static TensorFormat format = TensorFormat.NCHW;
	
	cudnnTensorDescriptor descriptor;
	
	private JCudnnDescriptor() {
		this.descriptor = new cudnnTensorDescriptor();
		checkError(JCudnn.cudnnCreateTensorDescriptor(this.descriptor));
	}
	
	JCudnnDescriptor(int[] dims) {
		this();
		int n = 1, c = 1, h = 1, w = 1;
		if(dims.length == 4) { // TODO: this arrangement is NCHW
			n = dims[0]; c = dims[1]; h = dims[2]; w = dims[3];
		}
		else if(dims.length == 2) {
			n = dims[0]; c = dims[1];
		}
		else if(dims.length == 1) { // TODO: is it right to use channel dimension for 1-dimensional tensor?
			c = dims[0];
		}
		else {
			throw new RuntimeException("Illegal dimensions for tensor descriptor: " + Arrays.toString(dims));
		}
		checkError(JCudnn.cudnnSetTensor4dDescriptor(this.descriptor, format.value(), dataType.value(), n, c, h, w));
	}
	
	JCudnnDescriptor(int[] dims, int channelStride) {
		this();
		if(dims.length != 4) {
			throw new RuntimeException("must have 4 axises to specify stride in tensor descriptor");
		}
		if(dims[1] > channelStride) {
			throw new RuntimeException(" channelStride " + channelStride + " is smaller than channel dimension " + dims[1]);
		}
		int n = dims[0], c = dims[1], h = dims[2], w = dims[3];
		int wStride = 1, hStride = wStride * w, cStride = hStride * h, nStride = cStride * channelStride;
		
		checkError(JCudnn.cudnnSetTensor4dDescriptorEx(this.descriptor, dataType.value(), n, c, h, w, nStride, cStride, hStride, wStride));
	}
	
	void free() {
		checkError(JCudnn.cudnnDestroyTensorDescriptor(this.descriptor));
	}
}

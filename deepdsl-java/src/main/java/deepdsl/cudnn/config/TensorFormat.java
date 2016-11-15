package deepdsl.cudnn.config;

import jcuda.jcudnn.cudnnTensorFormat;

public enum TensorFormat {
	NCHW(cudnnTensorFormat.CUDNN_TENSOR_NCHW),
	NHWC(cudnnTensorFormat.CUDNN_TENSOR_NHWC);
	
	private final int format;
    
    public int value() { return format; }
 
    TensorFormat(int format) {
		this.format = format;
	}
}

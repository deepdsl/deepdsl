package deepdsl.cudnn.config;

import jcuda.jcudnn.cudnnConvolutionMode;

public enum ConvolutionMode  {
	CROSS_CORRELATION (cudnnConvolutionMode.CUDNN_CROSS_CORRELATION), 
	CONVOLUTION (cudnnConvolutionMode.CUDNN_CONVOLUTION);
	
	private final int mode;
    
    public int value() { return mode; }
 
	ConvolutionMode(int mode) {
		this.mode = mode;
	}
}
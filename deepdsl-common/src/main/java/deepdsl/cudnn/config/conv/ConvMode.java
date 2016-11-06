package deepdsl.cudnn.config.conv;

import jcuda.jcudnn.cudnnConvolutionMode;

public enum ConvMode  {
	CROSS_CORRELATION (cudnnConvolutionMode.CUDNN_CROSS_CORRELATION), 
	CONVOLUTION (cudnnConvolutionMode.CUDNN_CONVOLUTION);
	
	private final int mode;
    
    public int mode() { return mode; }
 
	ConvMode(int mode) {
		this.mode = mode;
	}
}
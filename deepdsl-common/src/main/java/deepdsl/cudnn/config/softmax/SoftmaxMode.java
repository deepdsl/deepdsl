package deepdsl.cudnn.config.softmax;

import jcuda.jcudnn.cudnnSoftmaxMode;

public enum SoftmaxMode {
	INSTANCE(cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_INSTANCE),
	CHANNEL(cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_CHANNEL);
	
	private final int mode;
    
    public int mode() { return mode; }
 
    SoftmaxMode(int mode) {
		this.mode = mode;
	}
}

package deepdsl.cudnn.config.activation;

import jcuda.jcudnn.cudnnActivationMode;

public enum ActivationMode {
	CLIPPED_RELU(cudnnActivationMode.CUDNN_ACTIVATION_CLIPPED_RELU),
	RELU(cudnnActivationMode.CUDNN_ACTIVATION_RELU),
	SIGMOID(cudnnActivationMode.CUDNN_ACTIVATION_SIGMOID),
	TANH(cudnnActivationMode.CUDNN_ACTIVATION_TANH);
	
	private final int mode;
    
    public int mode() { return mode; }
 
    ActivationMode(int mode) {
		this.mode = mode;
	}
}

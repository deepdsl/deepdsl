package deepdsl.cudnn;
 
import deepdsl.util.ArithStats;
import jcuda.jcudnn.cudnnBatchNormMode; 
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationForwardInference;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationForwardTraining;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationBackward;

public class JCudnnBatchNorm extends JCudaFunction {
	static int mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL; // This is for convolution purpose
	static double epsilon = 1e-5; // I don't know what to pick so I picked the minimum.

	int forward_count = 0; 
	JCudnnDescriptor x_dptr, norm_dptr;  
	JCudaTensor running_mean, running_variance, saved_mean, saved_inv_variance;
	int[] x_dims, norm_dims;
	
	public JCudnnBatchNorm(int[] x_dims) {
		this.x_dims = x_dims;
		this.x_dptr = new JCudnnDescriptor(x_dims);
		this.norm_dims = new int[] {1, x_dims[1], 1, 1};
		this.norm_dptr = new JCudnnDescriptor(norm_dims);
		running_mean = new JCudaTensor(norm_dims);
		running_variance = new JCudaTensor(norm_dims);
		saved_mean = new JCudaTensor(norm_dims);
		saved_inv_variance = new JCudaTensor(norm_dims);
	}
	
	public void free() {
		this.x_dptr.free();
		this.norm_dptr.free();
		this.running_mean.free();
		this.running_variance.free();
		this.saved_inv_variance.free();
		this.saved_mean.free();
	}
	
	public JCudaTensor forward_inference(JCudaTensor x, JCudaTensor scale, JCudaTensor bias) {
		JCudaTensor y = new JCudaTensor(x_dims);
		
		int ret = cudnnBatchNormalizationForwardInference(cudnnHandle, mode, one, zero,
				x_dptr.descriptor, x.getData(), x_dptr.descriptor, y.getData(),
				norm_dptr.descriptor, scale.getData(), bias.getData(),
				running_mean.getData(), running_variance.getData(), epsilon);
		
		checkError(ret);
		
		return y;
	}
	
	public JCudaTensor forward(JCudaTensor x, JCudaTensor scale, JCudaTensor bias) {
		JCudaTensor y = new JCudaTensor(x_dims);

		double factor = 1.0/(1+forward_count++);
		
		long begin = System.nanoTime();
		
		int ret = cudnnBatchNormalizationForwardTraining(cudnnHandle, mode, one, zero, 
				x_dptr.descriptor, x.getData(), x_dptr.descriptor, y.getData(), 
				norm_dptr.descriptor, scale.getData(), bias.getData(), 
				factor, running_mean.getData(), running_variance.getData(), 
				epsilon, saved_mean.getData(), saved_inv_variance.getData()); 
		
		checkError(ret);
		
		ArithStats.cuda_timing("batch norm forward", begin);
		
		return y;
	}
	
	public JCudaTensor[] backward(JCudaTensor dy, JCudaTensor x, JCudaTensor scale) {
		JCudaTensor d_scale = new JCudaTensor(norm_dims), d_bias = new JCudaTensor(norm_dims);
		JCudaTensor dx = dy; // new JCudaTensor(x.getDims()); // This is in-place. Forward call cannot be in-place since use x
		long begin = System.nanoTime();
		
		int ret = cudnnBatchNormalizationBackward(cudnnHandle, mode, one, zero, one, zero,
				x_dptr.descriptor, x.getData(), x_dptr.descriptor, dy.getData(), x_dptr.descriptor, dx.getData(),
				norm_dptr.descriptor, scale.getData(), d_scale.getData(), d_bias.getData(), 
				epsilon, saved_mean.getData(), saved_inv_variance.getData());
		 
		checkError(ret);
		
		ArithStats.cuda_timing("batch norm backward", begin);
		
		return new JCudaTensor[]{dx, d_scale, d_bias};
	}
}

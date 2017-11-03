package deepdsl.cudnn;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable; 

import deepdsl.tensor.JTensor;
import deepdsl.tensor.JTensorFloat;
import deepdsl.util.ArithStats;
import jcuda.jcudnn.cudnnBatchNormMode; 
import jcuda.vec.VecFloat;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationForwardInference;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationForwardTraining;
import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationBackward;

class RunningMeanVariance implements Serializable { 
	private static final long serialVersionUID = -8414099972220055277L;
	int forward_count;
	JTensorFloat mean, variance;
	int dim;

	public RunningMeanVariance(int dim) {
		this(dim, 0, JTensor.constFloat(0, dim), JTensor.constFloat(0, dim));
	}

	public RunningMeanVariance(int dim, int forward_count, JTensorFloat mean, JTensorFloat variance) {
		this.dim = dim;
		this.forward_count = forward_count;
		this.mean = mean;
		this.variance = variance;
	}

	public RunningMeanVariance load(String name) {
		RunningMeanVariance ret = this;
		try {
			FileInputStream fileIn = new FileInputStream(name + ".ser");
			ObjectInputStream in = new ObjectInputStream(fileIn);
			RunningMeanVariance t = (RunningMeanVariance) in.readObject();
			in.close();
			fileIn.close();
			if(t != null) {
				if(this.dim == t.dim) {
					System.out.printf("Restored %s\n", name);
					ret = t; 
				}
			}
		}
		catch(IOException i) {
		}
		catch(ClassNotFoundException c) { 
		}
		return ret;
	}
	public void save(String name) {
		try {
			FileOutputStream fileOut = new FileOutputStream(name + ".ser");
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(this);
			out.close();
			fileOut.close();
			System.out.printf("Parameter is serialized in %s.ser\n", name);
		}
		catch(IOException i) {
			System.out.println(i);
		}
	}
}

public class JCudnnBatchNorm extends JCudaFunction {
	static int mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL; // This is for convolution purpose
	static double epsilon = 1e-5; // I don't know what to pick so I picked the minimum.

	int forward_count = 0; 
	JCudnnDescriptor x_dptr, norm_dptr;  
	JCudaTensor running_mean, running_variance, saved_mean, saved_inv_variance;
	int[] x_dims, norm_dims; 
	String path;

	boolean trained = false;

	JCudaTensor lowerBound, upperBound; // non-null only if variance clipping is used
	
	public static boolean withVarianceClipping = false;
	public static float CLIP_MULTIPLIER = 1.1f;
	public static int CLIP_START_ITER = 1000;

	// if fixed_factor <= 1/forward_count then use accumulative moving average else use exponential moving average
	public static double FIXED_FACTOR = 0.001; 
	public static boolean withRunningVariance = true; // whether to run inference with running mean/variance
	
	public JCudnnBatchNorm(String path, int[] x_dims) {
		this.x_dims = x_dims;
		this.path = path;
		this.x_dptr = new JCudnnDescriptor(x_dims);
		this.norm_dims = new int[] {1, x_dims[1], 1, 1};
		this.norm_dptr = new JCudnnDescriptor(norm_dims);

		RunningMeanVariance running = new RunningMeanVariance(x_dims[1]).load(path);
		running_mean = running.mean.asJCudaTensor();
		running_variance = running.variance.asJCudaTensor();
		forward_count = running.forward_count;

		saved_mean = new JCudaTensor(norm_dims);
		saved_inv_variance = new JCudaTensor(norm_dims);
		
		if(withVarianceClipping) {
			lowerBound = new JCudaTensor(norm_dims);
			upperBound = new JCudaTensor(norm_dims);
		}
	}
	@Override
	public void free() {
		this.x_dptr.free();
		this.norm_dptr.free();
		this.running_mean.free();
		this.running_variance.free();
		this.saved_inv_variance.free();
		this.saved_mean.free();

		if(withVarianceClipping) {
			this.lowerBound.free();
			this.upperBound.free();
		}
	}
	
	@Override
	public void save() {
		if (trained) {
			new RunningMeanVariance(x_dims[1], forward_count, 
					running_mean.asJTensor(), running_variance.asJTensor())
			.save(path);
		}
	}
	public JCudaTensor forward_inference(JCudaTensor x, JCudaTensor scale, JCudaTensor bias) {
		JCudaTensor ret;
		
		if(withRunningVariance) {
			ret = forward_inference_running_variance(x, scale, bias);
		}
		else {
			ret = forward_inference_no_running_variance(x, scale, bias);
		}
		return ret;
	}

	private JCudaTensor forward_inference_running_variance(JCudaTensor x, JCudaTensor scale, JCudaTensor bias) {
		JCudaTensor y = new JCudaTensor(x_dims);

		int ret = cudnnBatchNormalizationForwardInference(cudnnHandle, mode, one, zero,
				x_dptr.descriptor, x.getData(), x_dptr.descriptor, y.getData(),
				norm_dptr.descriptor, scale.getData(), bias.getData(),
				running_mean.getData(), running_variance.getData(), epsilon); 
		
		checkError(ret);

		return y;
	}

	// Use forward training. A little slower but for unknown reason works for ResNet.
	private JCudaTensor forward_inference_no_running_variance(JCudaTensor x, JCudaTensor scale, JCudaTensor bias) {
		JCudaTensor y = new JCudaTensor(x_dims);

		double factor = 0; // don't change running mean or variance

		int ret = cudnnBatchNormalizationForwardTraining(cudnnHandle, mode, one, zero, 
				x_dptr.descriptor, x.getData(), x_dptr.descriptor, y.getData(), 
				norm_dptr.descriptor, scale.getData(), bias.getData(), 
				factor, running_mean.getData(), running_variance.getData(), 
				epsilon, saved_mean.getData(), saved_inv_variance.getData());  

		checkError(ret);

		return y;
	}

	private double getFactor() { 
		double factor = 1.0 / (1 + forward_count++);
		
		return (FIXED_FACTOR > factor) ? FIXED_FACTOR : factor; 
	}
	
	public JCudaTensor forward(JCudaTensor x, JCudaTensor scale, JCudaTensor bias) {
		JCudaTensor y = new JCudaTensor(x_dims);

		double factor = getFactor();
		int channel = x_dims[1];
		
		long begin = System.nanoTime();

		if(withVarianceClipping) {
			if(forward_count > CLIP_START_ITER) {
				lowerBound = running_variance.clone(lowerBound).times_i((float) (1 - (1 - 1.0f / CLIP_MULTIPLIER) * factor));
				float average_variance = running_variance.sum() / channel;
				upperBound = running_variance.clone(upperBound).times_i((float) (1 + (CLIP_MULTIPLIER - 1) * factor)).plus_i((float) (average_variance * CLIP_MULTIPLIER * factor));
			}
		}
		int ret = cudnnBatchNormalizationForwardTraining(cudnnHandle, mode, one, zero, 
				x_dptr.descriptor, x.getData(), x_dptr.descriptor, y.getData(), 
				norm_dptr.descriptor, scale.getData(), bias.getData(), 
				factor, running_mean.getData(), running_variance.getData(), 
				epsilon, saved_mean.getData(), saved_inv_variance.getData()); 
		
		if(withVarianceClipping) {
			if(forward_count > CLIP_START_ITER) {
				// running_variance(n) / clip_multipler <= variance(n+1) <= (running_variance(n) + average(running_variance(n)) * clip_multipler
				VecFloat.fmin(channel, running_variance.getData(), running_variance.getData(), upperBound.getData());
				VecFloat.fmax(channel, running_variance.getData(), running_variance.getData(), lowerBound.getData());		
			}
		}
		checkError(ret);

		ArithStats.cuda_timing("batch norm forward", begin);

		trained = true;

		return y;
	}

	public JCudaTensor[] backward(JCudaTensor dy, JCudaTensor x, JCudaTensor scale) {
		JCudaTensor d_scale = new JCudaTensor(norm_dims), d_bias = new JCudaTensor(norm_dims);
		JCudaTensor dx = dy; // This is in-place. Forward call cannot be in-place since backward uses x
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

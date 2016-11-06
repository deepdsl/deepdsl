package deepdsl.cudnn;

import static jcuda.jcudnn.JCudnn.cudnnBatchNormalizationBackward;

import java.util.Arrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnTensorDescriptor;

import org.junit.Test;

import deepdsl.tensor.JTensorFloat;

public class TestBatchNorm {
	
	@Test
	public void forward() {
		float[] a = {1,2,3,4,5,6}, a1 = {10, 100, 1000}, a2 = {0.1f, 0.2f, 0.3f};
		int[] dims = {2, 3, 1, 1}, norm_dims = {1, 3, 1, 1};
		JTensorFloat t = new JTensorFloat(a, dims), 
				tscale = new JTensorFloat(a1, norm_dims), 
				tbias = new JTensorFloat(a2, norm_dims);
		JCudaTensor x = t.asJCudaTensor(),
				scale = tscale.asJCudaTensor(),
				bias = tbias.asJCudaTensor();
		
		JCudnnBatchNorm norm = new JCudnnBatchNorm(dims);
		
		JCudaTensor y = norm.forward(x, scale, bias);
		System.out.println(Arrays.toString(y.asArray()));
		System.out.println(Arrays.toString(norm.running_mean.asArray()));
		System.out.println(Arrays.toString(norm.running_variance.asArray()));
		System.out.println(Arrays.toString(norm.saved_mean.asArray()));
		System.out.println(Arrays.toString(norm.saved_inv_variance.asArray()));
		
		y = norm.forward_inference(x, scale, bias);
		System.out.println(Arrays.toString(y.asArray()));
	}
	
	// TODO: couldn't pass this test.
	@Test
	public void backward() {
		float[] a = {1, 2, 3, 4, 5 ,6}, a1 = {10, 20, 30}, a2 = {0.1f, 0.2f, 0.3f}, 
				a4 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
		int[] dims = {2, 3, 1, 1}, norm_dims = {1, 3, 1, 1};
		JTensorFloat t = new JTensorFloat(a, dims), 
				tscale = new JTensorFloat(a1, norm_dims), 
				tbias = new JTensorFloat(a2, norm_dims);
		JCudaTensor x = t.asJCudaTensor(),
				scale = tscale.asJCudaTensor(),
				bias = tbias.asJCudaTensor(), 
				dy = new JTensorFloat(a4, dims).asJCudaTensor();
		
		JCudnnBatchNorm norm = new JCudnnBatchNorm(dims);
		
		JCudaTensor y = norm.forward(x, scale, bias);
		 
		
		JCudaTensor dx[] = norm.backward(dy, x, scale);
		
		System.out.println("y: " + Arrays.toString(y.asArray()));
		System.out.println("dx: " + Arrays.toString(dx[0].asArray()));
		System.out.println("d_scale: " + Arrays.toString(dx[1].asArray()));
		System.out.println("d_bias: " + Arrays.toString(dx[2].asArray()));
		
		y = norm.forward(x, dx[1], dx[2]);
		System.out.println("y: " + Arrays.toString(y.asArray()));
		
		dx = norm.backward(dy, x, dx[1]);
		System.out.println("dx: " + Arrays.toString(dx[0].asArray()));
		System.out.println("d_scale: " + Arrays.toString(dx[1].asArray()));
		System.out.println("d_bias: " + Arrays.toString(dx[2].asArray()));
	}
	
	
	@Test
	public void test() {
		 cudnnHandle cudnnHandle = new cudnnHandle();
		 int mode = 1; // spatial mode
		 Pointer one = Pointer.to(new float[]{1.0f});
		 Pointer zero = Pointer.to(new float[]{0.0f});
		 int[] dims = {2, 3, 1, 1};
		 cudnnTensorDescriptor descriptor = new cudnnTensorDescriptor();
		 JCudnn.cudnnCreateTensorDescriptor(descriptor);
		 JCudnn.cudnnSetTensor4dDescriptor(descriptor, 0, 0, dims[0], dims[1], dims[2], dims[3]);
		 cudnnTensorDescriptor norm_dptr = new cudnnTensorDescriptor();
		 JCudnn.cudnnCreateTensorDescriptor(norm_dptr);
		 JCudnn.cudnnSetTensor4dDescriptor(norm_dptr, 0, 0, 1, dims[1], 1, 1);
		 double epsilon = 1;
		 
		 Pointer x = new Pointer(), dy = new Pointer(), dx = new Pointer();
		 int size = dims[0]*dims[1]*dims[2]*dims[3];
		 JCublas.cublasAlloc(size, Sizeof.FLOAT, x); 
		 JCublas.cublasAlloc(size, Sizeof.FLOAT, dy);
		 JCublas.cublasAlloc(size, Sizeof.FLOAT, dx);
		 JCublas2.cublasSetVector(size, Sizeof.FLOAT, Pointer.to(new float[] {1,2,3,4,5,6}), 1, x, 1);
		 int norm_size = dims[1];
		 Pointer scale = new Pointer(), d_scale = new Pointer(), d_bias = new Pointer(), saved_mean = new Pointer(), saved_variance = new Pointer();
		 JCublas.cublasAlloc(norm_size, Sizeof.FLOAT, scale); 
		 JCublas.cublasAlloc(norm_size, Sizeof.FLOAT, d_scale);
		 JCublas.cublasAlloc(norm_size, Sizeof.FLOAT, d_bias);
		 JCublas.cublasAlloc(norm_size, Sizeof.FLOAT, saved_mean);
		 JCublas.cublasAlloc(norm_size, Sizeof.FLOAT, saved_variance);
		 JCublas2.cublasSetVector(norm_size, Sizeof.FLOAT, Pointer.to(new float[] {10,100,1000}), 1, scale, 1);
		 
		 int ret = cudnnBatchNormalizationBackward(cudnnHandle, mode, one, zero, one, zero,
					descriptor, x, descriptor, dy, descriptor, dx,
					norm_dptr, scale, d_scale, d_bias, 
					epsilon, saved_mean, saved_variance);
			
			
		 System.out.println(ret); 
		 float[] result = new float[size];
		 JCublas2.cublasGetVector(size, Sizeof.FLOAT, dx, 1, Pointer.to(result), 1);
		 System.out.println(Arrays.toString(result));
	}
	
	
}

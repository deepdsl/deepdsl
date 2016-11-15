package deepdsl.tensor;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random; 

public abstract class JTensor implements Serializable { 
	private static final long serialVersionUID = 3474255477686160191L;
	public final int[] dim;  

	protected int columnIndex(int[] index) {
		int k = 0;
		for(int i=0; i<index.length; i++) { k = k*dim[i] + index[i]; } 
		return k;
	}
	protected int strideAt(int dimPosFromLeft) {
		int ret = 1; 
		for(int i=dimPosFromLeft; i<dim.length; i++) { ret *= dim[i]; }
		return ret;
	} 
	protected JTensor(int[] dim) { 
		this.dim = dim; 
	} 

	public static int size(int[] dim) {
		int size = 1;

		for(int i=0; i<dim.length; i++) {
			size *= dim[i];
		}
		return size;
	}
	public static JTensorFloat gaussianFloat(float std, int... dim) {
		int size = size(dim);

		float[] a = new float[size]; 

		for(int i=0; i<size; i++) { 
			a[i] = (float) new Random().nextGaussian() * std;
		} 
		JTensorFloat ret = new JTensorFloat(a, dim); 
		return ret;
	}
	public static JTensorFloat constFloat(float x, int...dim) {
		float[] a = new float[size(dim)];
		if (x != 0f) {
			Arrays.fill(a, x);
		}
		return new JTensorFloat(a, dim);
	}
	public static JTensorFloat zeroFloat(int...dim) {  
		return new JTensorFloat(new float[size(dim)], dim); 
	}
	public static JTensorFloat randomFloat (float low, float high, int... dim) {
		int size = size(dim);

		float[] a = new float[size];

		float range = high - low; 

		for(int i=0; i<size; i++) { 
			a[i] = low + new Random().nextFloat() * range;
		} 
		JTensorFloat ret = new JTensorFloat(a, dim); 
		return ret;
	}
}

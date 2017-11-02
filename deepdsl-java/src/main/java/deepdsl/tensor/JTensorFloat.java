package deepdsl.tensor; 

import deepdsl.cudnn.JCudaTensor;
import deepdsl.util.ArithStats;

import org.naokishibata.sleef.FastMath;

import java.io.*;
import java.util.Arrays;

public class JTensorFloat extends JTensor {  
	private static final long serialVersionUID = 346421286929171807L;  
	
	public final float[] array; 
	
	public JTensorFloat(float[] array, int[] dim) {
		super(dim);
		this.array = array; 
	}
	public int getLength() {
		int ret = 1;
		for(int d: dim) { ret *= d; }
		return ret;
	}
	public JTensorFloat load() {  
		return this;
	}
	public JTensorFloat load(String name) { 
		return _load(name);
	}
	public JCudaTensor loadCuda() { 
		return asJCudaTensor();
	}
	public JCudaTensor loadCuda(String name) { 
		return _load(name).asJCudaTensor();
	}
	private JTensorFloat _load(String name) {
		JTensorFloat ret = this; 
		try(FileInputStream fileIn = new FileInputStream(name + ".ser");
			ObjectInputStream in = new ObjectInputStream(fileIn)) 
		{
			JTensorFloat t = (JTensorFloat) in.readObject(); 
			if(t != null) {
				if(Arrays.toString(this.dim).equals(Arrays.toString(t.dim))) {
					if(t.getLength() == this.getLength()) {
						ret = t;
						System.out.printf("Restored %s\n", name); 
					}
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
			i.printStackTrace();
		}
	}
	
	// convert an array to an indicator matrix
	// [0, 2, 1] --> [[1,0,0], [0,0,1], [0,1,0]] if there are 3 classes
	public JTensorFloat asIndicator(int numOfCls) { 
		int length = this.getLength();
		float[] newArray = new float[length * numOfCls];
		
		for(int i=0; i<length; i++) {
			newArray[numOfCls * i + (int) this.array[i]] = 1;
		}
		int[] newDim = new int[this.dim.length + 1];
		for(int i=0; i<this.dim.length; i++) {
			newDim[i] = this.dim[i];
		}
		newDim[this.dim.length] = numOfCls; 
		return new JTensorFloat(newArray, newDim);
	}

	public JCudaTensor asJCudaTensor() {
		return new JCudaTensor(this);
	}
	
	public int[] prediction() { 
		if(dim.length != 2) {
			throw new RuntimeException(String.format("Error in calculating prediction. "
					+ "This tensor dim: %s. is not 2 dimensional", Arrays.toString(dim)));
		}
		return new JMatFloat(array, new int[] {dim[0]}).maxIndex();
	}
	
	// top "top" prediction accuracy 
	public float accuracy(JTensorFloat y_label, int top) {
		if(dim.length != 2 || y_label.dim.length != 1 || dim[0] != y_label.dim[0]) {
			throw new RuntimeException(String.format("Error in calculating precision. "
					+ "This tensor dim: %s. Y tensor dim: %s", Arrays.toString(dim), Arrays.toString(y_label.dim)));
		}
		
		int batch = dim[0];
		int k = dim[1]; 
		float[] copy = new float[k];
		float count = 0;
		
		for(int i=0; i<batch; i++) {

			System.arraycopy(array, i*k, copy, 0, k);
			float truth = copy[(int) y_label.array[i]];
			
			Arrays.sort(copy);
			
			for(int j=0; j<top; j++) { 
				if(copy[k - j - 1] == truth) {
					count = count + 1;
					break;
				}
			}
		}
		return count/batch;
	} 
	
	// for debugging 
	String print(int i) {
		i = i < getLength()? i : getLength();
		
		float[] a = new float[i];
		for(int k=0; k<i; k++) {
			a[k] = array[k];
		}
		return Arrays.toString(a);
	}
	
	public String toString() {
		return Arrays.toString(array);
	}
	
	private float[] getArray() { return array; }
	private float[] getArrayCopy() { 
		float[] ret = new float[getLength()];
		System.arraycopy(array, 0, ret, 0, getLength());
		return ret;
	}  

	public JTensorFloat clone() { return new JTensorFloat(getArrayCopy(), dim); }
	
	public JTensorFloat update(JTensorFloat that, float alpha, float beta) {
		int size = getLength(), thatSize = that.getLength();
		if(size != thatSize) {
			throw new RuntimeException("tensor sizes do not match");
		}  
		for(int i=0; i<size; i++) {
			array[i] = array[i] * beta + that.array[i]* alpha;
		}

		return this;
	}
	
	public float get(int... index) { return array[columnIndex(index)]; } 
	
	public JTensorFloat slice(int... index) {
		long begin = System.nanoTime();
		int stride = strideAt(index.length);
		float[] a = new float[stride];
		System.arraycopy(array, columnIndex(index)*stride, a, 0, stride);
		
		int[] d = new int[dim.length - index.length];
		
		for(int i=index.length; i<dim.length; i++) {
			d[i-index.length] = dim[i];
		}

		ArithStats.timing("slice", begin);
		return new JTensorFloat(a, d);
	}
	
	public JTensorFloat call(FunFloat f) {
		float[] a = getArray();
		for(int i=0; i<a.length; i++) {
			a[i] = f.apply(array[i]);
		}
		return new JTensorFloat(a, dim);
	}

	public float max() {
		float ret = array[0];
		for(int i=1; i<getLength(); i++) {
			if (ret < array[i]) {
				ret = array[i];
			}
		}
		return ret;
	}
	
	// dot product
	public float dot(JTensorFloat t) {
		long begin = System.nanoTime();
		float ret = 0;
		if(t.getLength() != getLength()) { throw new RuntimeException("tensor array lengths do not match"); }

		for(int i=0; i<getLength(); i++) {
			ret += array[i] * t.array[i];
		}
		ArithStats.timing("dot_prod", begin);
		return ret;
	}
	
	public JTensorFloat flatten(int start, int... mid_dim) { 
		int middle = 1;
		for(int i=0; i<mid_dim.length; i++) {
			middle = middle*mid_dim[i];
		}
		int[] d = new int[dim.length - mid_dim.length + 1];
		for(int i=0; i<start; i++) {
			d[i] = dim[i];
		}
		d[start] = middle;
		for(int i=start + mid_dim.length; i<dim.length; i++) {
			d[i-mid_dim.length+1] = dim[i];
		} 
		return new JTensorFloat(array, d);
	}
	
	public JTensorFloat unflatten(int start, int... mid_dim) { 
		int[] d = new int[dim.length + mid_dim.length - 1];
		for(int i=0; i<start; i++) {
			d[i] = dim[i];
		}
		for(int i=start; i<start+mid_dim.length; i++) {
			d[i] = mid_dim[i-start];
		}
		for(int i=start+mid_dim.length; i<d.length; i++) {
			d[i] = dim[i-mid_dim.length+1];
		} 
		return new JTensorFloat(array, d);
	}
	  
	public JTensorFloat plus(JTensorFloat t) {
		long begin = System.nanoTime();
		if(getLength() != t.getLength()) {
			throw new RuntimeException("tensor lengths do not match"); 
		}

		float[] a = getArray();
		for(int i=0; i<getLength(); i++) {
			a[i] = array[i] + t.array[i];
		} 

		ArithStats.timing("plus", begin);
		return new JTensorFloat(a, dim);
	}
	// this * x
	public JTensorFloat times(float x) {
		long begin = System.nanoTime();
		float[] a = getArray();
		for(int i=0; i<getLength(); i++) {
			a[i] = array[i] * x;
		}
		ArithStats.timing("times-scalar", begin);
		return new JTensorFloat(a, dim);
	}
	// element-wise product
	public JTensorFloat times(JTensorFloat t) {
		long begin = System.nanoTime();
		if(getLength() != t.getLength()) { throw new RuntimeException("tensor lengths do not match"); }
		float[] a = getArray();
		for(int i=0; i<getLength(); i++) {
			a[i] = array[i] * t.array[i];
		}
		ArithStats.timing("elem-wise_prod", begin);
		return new JTensorFloat(a, dim);
	}
	public float sum() {
		long begin = System.nanoTime();
		float ret = 0;
		for(int i=0; i<getLength(); i++) {
			ret += (array[i]);
		}
		ArithStats.timing("sum", begin);
		return ret;
	}
	public JTensorFloat tanh() {
		long begin = System.nanoTime();
		float[] a = getArray();
		for(int i=0; i<getLength(); i++) {
			a[i] = FastMath.tanhf(array[i]);
		}
		ArithStats.timing("tanh", begin);
		return new JTensorFloat(a, dim);
	}
	public JTensorFloat log() {
		long begin = System.nanoTime();
		float[] a = getArray();
		for(int i=0; i<getLength(); i++) {
			a[i] = FastMath.logf(array[i]);
		}
		ArithStats.timing("log", begin);
		return new JTensorFloat(a, dim);
	}
	public JTensorFloat exp() {
		long begin = System.nanoTime();
		float[] a = getArray();
		for(int i=0; i<getLength(); i++) {
			a[i] = FastMath.expf(array[i]);
		}
		ArithStats.timing("exp", begin);
		return new JTensorFloat(a, dim);
	}
	public JTensorFloat indicator(JTensorFloat t) {
		long begin = System.nanoTime();
		if(getLength() != t.getLength()) { throw new RuntimeException("tensor lengths do not match"); }
		float[] a = getArray();
		for(int i=0; i<getLength(); i++) {
			a[i] = (array[i] == t.array[i])? 1 : 0;
		}
		ArithStats.timing("indicator", begin);
		return new JTensorFloat(a, dim);
	}
	public JTensorFloat copy(int... moreDim) {
		long begin = System.nanoTime();
		int size = 1;
		for(int i=0; i<moreDim.length; i++) {
			size *= moreDim[i];
		}
		float[] a = new float[size* getLength()];
		for(int i=0; i<size; i++) {
			System.arraycopy(array, 0, a, i*getLength(), getLength());
		}
		int[] d = new int[dim.length + moreDim.length];
		System.arraycopy(dim, 0, d, 0, dim.length);
		System.arraycopy(moreDim, 0, d, dim.length, moreDim.length);
		ArithStats.timing("copy", begin);
		return new JTensorFloat(a, d);
	}
	public JTensorFloat pow(float x) {
		long begin = System.nanoTime();
		int size = getLength();
		float[] a = getArray();
		for(int i=0; i<size; i++) {
			 a[i] = FastMath.powf(array[i], x);
		}
		ArithStats.timing("pow", begin);
		return new JTensorFloat(a, dim);
	}
}

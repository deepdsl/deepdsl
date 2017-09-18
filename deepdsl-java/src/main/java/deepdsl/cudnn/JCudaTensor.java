package deepdsl.cudnn;
 
import deepdsl.tensor.JTensorFloat;
import deepdsl.util.ArithStats;
import jcuda.Pointer;
import jcuda.Sizeof; 
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;  

import java.util.*; 
 
//import jcuda.runtime.JCuda;
import jcuda.vec.VecFloat;
import static deepdsl.cudnn.JCudaFunction.*;

class MemoryManager {
	private class Pair {
		int size; Pointer data;
		Pair(int size, Pointer data) { this.size = size; this.data = data; }
	}

	List<Pair> size_pointer_pairs = new ArrayList<Pair>();
	Map<Pointer, Integer> pointerSizes = new HashMap<>();
	long total = 0;
	
	void free(Pointer data) {
		if(! pointerSizes.containsKey(data))
			throw new RuntimeException("freeing unknown pointer");	
		size_pointer_pairs.add(new Pair(pointerSizes.get(data), data));
	} 
	
	Pointer alloc(int size) {
		Pointer data;
		Pair p = null;

		for(Pair pair : size_pointer_pairs) {
			if (pair.size == size) {
				p = pair;
			}  
		}
		if (p == null) {
			for(Pair pair: size_pointer_pairs) {
				if (pair.size >= size && size >= pair.size * 0.7) { 
					p = pair; 
				} 
			}
		}
		if (p != null) {
			data = p.data;
			size_pointer_pairs.remove(p);
		}
		else {
			data = new Pointer(); 
			
			JCudaFunction.allocFloat(data, size);
			
			pointerSizes.put(data,  size);
			total += size * 4;
		}
		return data;
	}
	void clear() {
		for(Pair pair: size_pointer_pairs) {
			JCudaFunction.free(pair.data);
		}
		size_pointer_pairs.clear();
	}
}


public class JCudaTensor {
	private Pointer data;
	private final int[] dims;
	final int size;
	private final static cublasHandle handle = cublasHandle;  

	public JCudaTensor(Pointer data, int[] dims) {
		this.data = data; 
		this.dims = dims;
		this.size = size();
	}

	static MemoryManager memory = new MemoryManager();
	public static boolean useCache = false;   
	 
	public JCudaTensor(int[] dims) {
		this(! useCache ? new Pointer() : memory.alloc(size(dims)), dims);

		long begin = System.nanoTime();
		if (! useCache) {
			allocFloat(data, size);
		}  
		ArithStats.cuda_timing("JCudaTensor allocation", begin);
	}

	public static void clearMemoryCache() { memory.clear(); }
	
	public static void enableMemoryCache() { useCache = true; } 
	
	public static void disableMemoryCache() { useCache = false; }
	
	public static void enableWorkspaceCache() { JCudaFunction.enableWorkspaceCache(); }
	
	public JCudaTensor(JTensorFloat tensor) {
		this(tensor.dim);
		long begin = System.nanoTime();
		
		copyHostToDevice(tensor.array, data); 
		 
		ArithStats.cuda_timing("JCudaTensor (copy-in)", begin);
	}

	public void save(String name) {
		this.asJTensor().save(name);
	}
	public int[] prediction() { 
		return asJTensor().prediction();
	}
	public float accuracy(JTensorFloat y_label, int top) {
		return asJTensor().accuracy(y_label, top);
	}
	
	public JCudaTensor clone() { return clone(new JCudaTensor(this.dims)); }
	
	public JCudaTensor clone(JCudaTensor ret) { 
		long begin = System.nanoTime();
		
		copyDeviceToDevice(size, data, ret.data); 

		ArithStats.cuda_timing("cloning", begin);
		return ret;
	}

	private int size() {
		return size(this.dims);
	}

	public static int size(int[] dims) {
		int size = 1;
		for(int i=0; i<dims.length; i++) { size = size * dims[i]; }
		return size;
	}

	// if rowMajor,    then split at nth dimension (0 .. n-1) as rows and (n .. dims.length-1) as columns)
	// if columnMajor, then split at nth dimension (n .. dims.length-1) as rows and (0 .. n-1) as columns
	public JCudaMatrix asMatrix(int n, boolean rowMajor) {
		if(n < 0 || n >=  dims.length) {
			throw new RuntimeException("cannot split beyond the index range");
		}
		int[] new_dims;
		int numOfColumns = 1;

		if(rowMajor) {
			new_dims = new int[n];
			for(int i=0; i<n; i++) {
				new_dims[i] = dims[i];
			}
			for(int i=n; i<dims.length; i++) {
				numOfColumns = numOfColumns * dims[i];
			}
		} else {
			new_dims = new int[dims.length - n];
			for(int i=n; i<dims.length; i++) {
				new_dims[i-n] = dims[i];
			}
			for(int i=0; i<n; i++) {
				numOfColumns = numOfColumns * dims[i];
			}
		}

		return new JCudaMatrix(data, new_dims, numOfColumns, rowMajor);
	}

	// cast to matrix
	public JCudaMatrix asMatrix() {
		if(dims.length > 2) {
			throw new RuntimeException("only 1 or 2 dimensional tensor can be cast to matrix");
		} 
		return dims.length == 2 ? asMatrix(1, true) : asVector() ;
	}

	// cast to column vector
	public JCudaMatrix asVector() { return asMatrix(0, false); }

	public void free() { 
		long begin = System.nanoTime();
 
		if (! useCache) { // when memory reuse is disabled
			JCudaFunction.free(data); 
		}
		else {
			memory.free(data);
		}
		ArithStats.cuda_timing("cuda free", begin);
	} 
	
// NOT used
	
//	public static float memoryInfo() {
//		long[] free = new long[1], total = new long[1];
//		JCuda.cudaMemGetInfo(free, total);
//		currentFree = free[0];
//		if(free[0] < minFree) {
//			minFree = free[0];
//		}
//		if(total[0] > maxTotal) {
//			maxTotal = total[0];
//		}
//		return (total[0] - free[0])/1000000f;
//	}
//
//	public static long minFree = Long.MAX_VALUE, maxTotal = 0, currentFree;
//	public static long MB = (long) Math.pow(10, 6);
//
//	public static String memoryUsage() {
//		return "current: " + (maxTotal - currentFree)/MB + " max: " + (maxTotal - minFree)/MB;
//	}

	public float[] asArray() {  
		float[] ret = new float[size];
		copyDeviceToHost(data, ret); 
		return ret;
	}

	public JTensorFloat asJTensor() {
		return new JTensorFloat(asArray(), dims);
	}

	public float dot(JCudaTensor that) {
		long begin = System.nanoTime();
		float[] result = new float[1];
		
		int ret = JCublas2.cublasSdot(handle, size, this.data, 1, that.data, 1, Pointer.to(result)); 
		checkError(ret);
		
		ArithStats.cuda_timing("cuda dot", begin);
		return result[0];
	}

	public float sum() {
		float[] result = new float[1];
		int n = size, incx = 1;
		Pointer x = this.data;
		
		int ret = JCublas2.cublasSasum(handle, n, x, incx, Pointer.to(result)); 
		checkError(ret);
		
		return result[0];
	}

	// making n copies and return n x dims tensor
	public JCudaTensor copy(int n) { return asVector().copy(n); }
	
	public JCudaTensor copy(int n, JCudaTensor ret) { return asVector().copy(n, ret); }

	// this = this * beta + that * alpha
	public JCudaTensor update(JCudaTensor that, float alpha, float beta) { 
		if(size != that.size) {
			throw new RuntimeException("tensor sizes do not match");
		}
		asVector().update(that.asVector(), alpha, beta); 
		return this;
	}

	// this = this * scale
	public JCudaTensor times_i(float scale) {
		long begin = System.nanoTime(); 
		
		if(scale != 1) {
			Pointer x = getData();
			VecFloat.mulScalar(size, x, x, scale);
		}
		ArithStats.cuda_timing("cuda times scalar", begin);
		return this;
	}

	// point-wise product
	public JCudaTensor times_i(JCudaTensor that) {
		long begin = System.nanoTime(); 
		int n = size;
		if(n != that.size) {
			throw new RuntimeException("Unmatched sizes in point-wise product: " + n + " != " + that.size);
		}
		Pointer x = getData(), y = that.getData();
		VecFloat.mul(size, x, x, y);
		 
		ArithStats.cuda_timing("cuda times tensor", begin);
		return this;
	}

	// this = this + that
	public JCudaTensor plus_i(JCudaTensor that) {
		long begin = System.nanoTime(); 
		Pointer x = getData(), y = that.getData();
		VecFloat.add(size, x, x, y);
		 
		ArithStats.cuda_timing("cuda add tensor", begin);
		return this;
	}

	// this = this + y
	public JCudaTensor plus_i(float y) {
		long begin = System.nanoTime(); 
		
		if(y != 0) {
			Pointer x = getData();
			VecFloat.addScalar(size, x, x, y);
		}
		ArithStats.cuda_timing("cuda add tensor to scalar", begin);
		return this;
	}
	
	public JCudaTensor log() {
		long begin = System.nanoTime(); 
		Pointer x = getData();
		VecFloat.log(size, x, x);
		 
		ArithStats.cuda_timing("cuda tensor log", begin);
		return this;
	}

	public JCudaTensor pow(float y) {
		long begin = System.nanoTime(); 
		Pointer x = getData();
		
		if(y == -1) {
			VecFloat.scalarDiv(size, x, 1, x);
		}
		else {
			VecFloat.pow(size, x, x, JCudaMatrix.constantVector(size, 1, y).data); // TODO: improve this
		}
		ArithStats.cuda_timing("cuda tensor pow", begin);
		return this;
	}

	// used to implement indicator vector
	public JCudaTensor eq(JCudaTensor that) {
		Pointer x = getData(), y = that.getData();

		VecFloat.eq(size, x, x, y);

		return this;
	}
 
	public JCudaTensor clip(float limit) {
		long begin = System.nanoTime(); 
		Pointer x = getData();  
		
		reserveWorkspace(size * Sizeof.FLOAT);
		Pointer y = getWorkspace();
		
		VecFloat.set(size, y, limit);
		VecFloat.fmin(size, x, x, y); 
		VecFloat.set(size, y, -limit);
		VecFloat.fmax(size, x, x, y);
		 
		ArithStats.cuda_timing("cuda clip tensor", begin);
		return this;
	}

	// flatten tensor dims from index "start" to "start + dims.length". 
	// that is to replace these dimensions with the product of "dims"
	public JCudaTensor flatten(int start, int[] dims) { 
		int dim = 1;
		// checking consistency and throw exception if not matched.
		for(int i=start; i<dims.length+start; i++) {
			if(this.dims[i] != dims[i-start]) {
				throw new RuntimeException("dims do not match in flattened tensor");
			}
			dim = dim * dims[i-start];
		}
		int[] newDims = new int[this.dims.length - dims.length + 1];
		for(int i=0; i<start; i++) {
			newDims[i] = this.dims[i];
		}
		newDims[start] = dim;
		for(int i=start+1; i<newDims.length; i++) {
			newDims[i] = this.dims[i+dims.length-1];
		}
		return new JCudaTensor(data, newDims);
	}
	// unflatten tensor dim at "start" and replace it with "dims"
	public JCudaTensor unflatten(int start, int[] dims) { 
		int dim = 1;
		// checking consistency and throw exception if not matched.
		for(int i=0; i<dims.length; i++) {	
			dim = dim * dims[i];
		}
		if(this.dims[start] != dim) {
			throw new RuntimeException("dims do not match in unflattened tensor");
		}
		int[] newDims = new int[this.dims.length + dims.length - 1];
		for(int i=0; i<start; i++) {
			newDims[i] = this.dims[i];
		}
		for(int i=start; i<start + dims.length; i++) {
			newDims[i] = dims[i-start];
		}
		for(int i=start+dims.length; i<newDims.length; i++) {
			newDims[i] = this.dims[i-dims.length+1];
		}
		return new JCudaTensor(data, newDims);
	} 

	public int length() { 
		int dimProd = 1;
		for (int dim : dims) {
			dimProd *= dim;
		}
		return dimProd;
	}

	public int[] getDims() {
		return dims;
	}

	public Pointer getData() { 
		return data;
	}
	
	public Pointer getData(int offset) {
		return data.withByteOffset(offset * 4);
	}
}

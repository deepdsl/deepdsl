package deepdsl.util;

import java.io.File; 
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;

public abstract class CudaRun { 
	String network_dir;    
	JTensorFactory trainData, testData;

	protected Set<JCudaFunction> layers = new HashSet<>(); // layers
	protected Map<String, JCudaTensor> params = new HashMap<>();   // parameters 
	private Set<String> fixedParams = new HashSet<>();

	public CudaRun(String path) {
		this.network_dir = path; 

		File f = new File(path);
		if(!f.exists()) { 
			if(!f.mkdir()) {
				throw new RuntimeException("cannot create the network directory: " + path); 
			}
		}
		JCudaTensor.enableMemoryCache();
		JCudaFunction.enableWorkspaceCache();
	}

	public void setTrainData(JTensorFactory factory) { trainData = factory; }
	public void setTestData(JTensorFactory factory) { testData = factory; }

	private <X extends JCudaFunction> X add(X l) {
		layers.add(l);
		return l;
	}

	public JCudnnConvolution addConvolution(int[] x, int[] w, int[] b, int stride, int padding) {
		return add(new JCudnnConvolution(x, w, b, stride, padding)); 
	}
	public JCudnnDropout addDropout(String key, int[] x, float dropout) {
		return add(new JCudnnDropout(network_dir+"/"+key, x ,dropout));
	}
	public JCudnnLRN addLRN(int[] x, int n, double alpha, double beta) {
		return add(new JCudnnLRN(x, n, alpha, beta));
	}
	public JCudnnSoftmax addSoftmax(int[] x, SoftmaxAlgorithm algorithm) {
		return add(new JCudnnSoftmax(x, algorithm));
	}
	public JCudnnPooling addPooling(int[] x, int window, int stride, int padding, PoolingType poolType) {
		return add(new JCudnnPooling(x, window, stride, padding, poolType));
	}
	public JCudnnActivation addActivation(int[] x, ActivationMode mode) {
		return add(new JCudnnActivation(x, mode));
	}
	public JCudnnConcat addConcat(int[]... dimsList) {
		return add(new JCudnnConcat(dimsList));
	}
	public JCudnnBatchNorm addBatchNorm(String key, int[] x) {
		return add(new JCudnnBatchNorm(network_dir+"/"+key, x));
	}

	public JCudaTensor addParam(String key, String type, float x, int ... dim) {
		return addParam(key, false, type, x, dim);
	}
	public JCudaTensor addFixedParam(String key, String type, float x, int ... dim) {
		return addParam(key, true, type, x, dim);
	}
	private JCudaTensor addParam(String key, boolean isFixed, String type, float x, int ... dim) {
		String path = network_dir+"/"+key+".ser";

		JCudaTensor ret;
		if(isFixed) {
			ret = init(type, x, dim).loadCuda();
			fixedParams.add(key);
		}
		else if(Files.exists(Paths.get(path))) { 
			ret = new JTensorFloat(null, dim).loadCuda(network_dir+"/"+key);
		}
		else {
			ret = init(type, x, dim).loadCuda();
		}
		params.put(key, ret);
		return ret;
	} 

	private JTensorFloat init(String type, float x, int[] dim) {
		JTensorFloat t;
		switch(type) {
		case "Constant" : 
			if(x == 0) {
				t = JTensor.zeroFloat(dim);
			}
			else {
				t = JTensor.constFloat(x, dim); 
			}
			break;
		case "Random" : 
			t = JTensor.randomFloat(-x, x, dim);
			break;
		case "Gaussian" : 
			t = JTensor.gaussianFloat(x, dim);
			break;
		default:
			throw new RuntimeException("Incorrect parameter type: " + type);	
		}
		return t;
	}

	public void free() {
		for(JCudaFunction f: layers) { f.free(); }
		for(String k: params.keySet()) { params.get(k).free(); }
		layers.clear();
		params.clear();
		fixedParams.clear();
		JCudaTensor.clearMemoryCache();
		JCudaFunction.destroy();
	}

	public void save() {
		for(JCudaFunction f: layers) { f.save(); }
		for(String k: params.keySet()) {  
			if(!fixedParams.contains(k)) {
				params.get(k).save(network_dir+"/"+k);
			} 
		}
	}

	public abstract float trainFunction(JTensorFloat x, JTensorFloat y);
	public abstract JCudaTensor testFunction(JTensorFloat x);

	public void train(int train_itr) {
		double start = System.nanoTime();
		for(int i=1; i<=train_itr; i++) {
			JTensorFloatTuple t = trainData.nextFloat();
			float loss = trainFunction(t.getImage(), 
					t.getLabel());
			System.out.println(i + " " + loss);
		}
		System.out.println((System.nanoTime()-start)/1E9);
	} 

	public void test(int test_itr) {
		float average = 0;
		for(int i=1; i<=test_itr; i++) {
			JTensorFloatTuple t = testData.nextFloat();
			float precision = testFunction(t.getImage()).accuracy(t.getLabel(), 1);
			average += precision;
			System.out.println(i + " " + precision);
		}
		System.out.println("Average precision: " + average/test_itr);
	}

	public void infer(int test_itr) {
		for(int i=1; i<=test_itr; i++) {
			JTensorFloatTuple t = testData.nextFloat();
			int[] p = testFunction(t.getImage()).prediction(); 
			System.out.println(i + " " + Arrays.toString(p));
		} 
	}
}

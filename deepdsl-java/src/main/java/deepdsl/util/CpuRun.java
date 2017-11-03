package deepdsl.util;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;



import deepdsl.tensor.JTensor;
import deepdsl.tensor.JTensorFactory;
import deepdsl.tensor.JTensorFloat;
import deepdsl.tensor.JTensorFloatTuple;

public abstract class CpuRun {
	String network_dir;    
	JTensorFactory trainData, testData;

	protected Map<String, JTensorFloat> params = new HashMap<>();   // parameters 
	private Set<String> fixedParams = new HashSet<>();

	public CpuRun(String path) {
		this.network_dir = path; 

		File f = new File(path);
		if(!f.exists()) { 
			if(!f.mkdir()) {
				throw new RuntimeException("cannot create the network directory: " + path); 
			}
		} 
	}

	public void setTrainData(JTensorFactory factory) { trainData = factory; }
	public void setTestData(JTensorFactory factory) { testData = factory; }

	public JTensorFloat addParam(String key, String type, float x, int ... dim) {
		return addParam(key, false, type, x, dim);
	}
	public JTensorFloat addFixedParam(String key, String type, float x, int ... dim) {
		return addParam(key, true, type, x, dim);
	}
	private JTensorFloat addParam(String key, boolean isFixed, String type, float x, int ... dim) {
		String path = network_dir+"/"+key+".ser";

		JTensorFloat ret;
		if(isFixed) {
			ret = init(type, x, dim);
			fixedParams.add(key);
		}
		else if(Files.exists(Paths.get(path))) { 
			ret = new JTensorFloat(null, dim).load(network_dir+"/"+key);
		}
		else {
			ret = init(type, x, dim);
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
		params.clear();
	}

	public void save() {
		for(String k: params.keySet()) {  
			if(!fixedParams.contains(k)) {
				params.get(k).save(network_dir+"/"+k);
			} 
		}
	}

	public abstract float trainFunction(JTensorFloat x, JTensorFloat y);
	public abstract JTensorFloat testFunction(JTensorFloat x);

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

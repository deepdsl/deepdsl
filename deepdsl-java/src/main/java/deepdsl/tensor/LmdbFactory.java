package deepdsl.tensor;
 
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

import org.fusesource.lmdbjni.Database;
import org.fusesource.lmdbjni.Entry;
import org.fusesource.lmdbjni.Env;
import org.fusesource.lmdbjni.Transaction; 

import deepdsl.data.imagenet.LmdbUtils;
import deepdsl.data.imagenet.Tuple;
import deepdsl.util.ArithStats;

public class LmdbFactory extends JTensorFactory {
	Iterator<Entry> iterator;
	Transaction tx;
	Database db;
	int max_size;
	int count = 0;  
	int[] mean; 
	boolean centerCropping = false;
 
	private LmdbFactory(int[] dims, String path, int max_size, int numOfClasses, boolean centerCropping) {
		super(dims);   
		this.centerCropping = centerCropping;
		mean = (dims[1] == 3) ? new int[]{104, 117, 123} : new int[dims[1]]; // use fixed mean if channel size is 3  
		
		Env env = LmdbUtils.getEnv(path);
		tx = env.createReadTransaction();
		db = env.openDatabase();
		iterator = LmdbUtils.getIterator(LmdbUtils.getIterable(tx, db)); 
		
		this.max_size = max_size;

		if(max_size < batch) {
			throw new RuntimeException("not enough training/test samples for even one batch");
		} 
		if(_k == 0) {
			_k = numOfClasses;
			if(labelMap == null || labelMap.length != _k) {
				labelMap = new int[numOfClasses];
				Arrays.fill(labelMap, -1);
			}
		}
		else if(_k != numOfClasses){
			throw new RuntimeException("Different number of classes for training/test samples");
		}
	} 
	
	// The lookup method is for testing only to force the class label to have a number of classes less than 1000
	static final int maxNumOfClasses = 1000;
	static int _k = 0;
	static int[] labelMap = {365, 981, 863, 342, 819, 337, 374, 608, 895, 263}; // hard-coded class label translation for internal testing
	
	static int lookup(int x) {
		if(_k >= maxNumOfClasses) { // don't translate labels if there are already 1000 classes
			return x;
		}
		for(int i=0; i<_k; i++) {
			if(labelMap[i] == x) {
				return i;
			}
			else if (labelMap[i] == -1) {
				labelMap[i] = x;
				return i;
			}
		}
		throw new RuntimeException(String.format("Data has more than %d classes of labels", _k));
	}

	public JTensorFloatTuple nextFloat() {
		long begin = System.nanoTime(); 

		count = count + 1;

		if (count * batch > max_size) {
			//In this case, just reset iterator to the beginning by getting a fresh one
			iterator = LmdbUtils.getIterator(LmdbUtils.getIterable(tx, db)); 
			count = 1;
		} 

		int batch = dims[0], channel = dims[1], height = dims[2], width = dims[3];
		float[] images = new float[size];
		float[] labels = new float[batch]; 
		
		int x = 0;
		
		for (int i = 0; i < batch; i ++) {
			Entry entry = iterator.next();
			Tuple tuple = LmdbUtils.nextTuple(entry);
			byte[] bytes = tuple.image;
			int label = tuple.label;
			int[] image_dims = tuple.dims;
			 
			if(image_dims[0] != channel || image_dims[1] < height || image_dims[2] < width) {
				throw new RuntimeException(String.format("image dimension %s is not compatible with specified dimensions (last 3) %s", 
						Arrays.toString(image_dims), Arrays.toString(dims)));
			}
			
			int stride = image_dims[1] * image_dims[2];
			
			if(image_dims[1] == height && image_dims[2] == width) {
				for(int k = 0; k < channel; k++) {
					int channel_offset = k * stride; 

					for(int j = 0; j < stride; j++) {  
						images[x++] = ((int) (bytes[channel_offset + j] & 0xFF) - mean[k])/128.0f;
					}
				}
			} 
			else {  
				int height_index = centerCropping ? (image_dims[1] - height)/2 : new Random().nextInt(image_dims[1] - height + 1);
				int width_index = centerCropping? (image_dims[2] - width)/2 : new Random().nextInt(image_dims[2] - width + 1); 
 
				for(int k = 0; k < channel; k++) {
					int channel_offset = k * stride; 

					for(int h = 0; h < height; h++) { 
						int height_offset = channel_offset + (height_index + h) * image_dims[2] + width_index;

						for(int w = 0; w < width; w++) {
							images[x++] = ((int) (bytes[height_offset + w] & 0xFF) - mean[k])/128.0f;
						}

					}
				}
			}
			labels[i] = lookup(label); // FIXME: Users can replace "lookup(label)" with just "label" for actual training with 1000 classes
		} 

		JTensorFloatTuple pair = new JTensorFloatTuple(new JTensorFloat(images, dims), 
														new JTensorFloat(labels, new int[]{batch}));
		ArithStats.timing("sliceLmdbFloat", begin); 
		
		return pair;
	}

	/**
	 * The method to set up Factory for stream loading from LMDB
	 *
	 * @param dims
	 * @param path the path of the LMDB data 
	 * @return
	 */ 
	public static LmdbFactory getFactory(String path, int max_size, int[] dims, int numOfClasses, boolean centerCropping) {
		return new LmdbFactory(dims, path, max_size, numOfClasses, centerCropping);
	}
	public static LmdbFactory getFactory(String path, int max_size, int[] dims, int numOfClasses) {
		return getFactory(path, max_size, dims, numOfClasses, false);
	}
 
	public static LmdbFactory getFactory(String path, int max_size, int[] dims) {
		return getFactory(path, max_size, dims, 10);
	}
	 
	 
	public String toString() { return "LMDB " + Arrays.toString(dims); }
}

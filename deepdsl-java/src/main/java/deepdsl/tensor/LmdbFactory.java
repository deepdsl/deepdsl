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
	LmdbUtils.OS os; 
	int max_size;
	int count = 0;  
	int[] mean; 
 
	private LmdbFactory(int[] dims, String path, LmdbUtils.OS os, int max_size) {
		super(dims);   
		mean = (dims[1] == 3) ? new int[]{104, 117, 123} : new int[dims[1]]; // use fixed mean if channel size is 3  
		
		Env env = LmdbUtils.getEnv(path);
		tx = env.createReadTransaction();
		db = env.openDatabase();
		iterator = LmdbUtils.getIterator(LmdbUtils.getIterable(tx, db));
		this.os = os;
		this.max_size = max_size;

		if(max_size < batch) {
			throw new RuntimeException("not enough training/test samples for even one batch");
		}
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
			Tuple tuple = LmdbUtils.nextTuple(entry, os);
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
				// TODO: center the cropping for testing
				int height_index = new Random().nextInt(image_dims[1] - height + 1);
				int width_index = new Random().nextInt(image_dims[2] - width + 1); 
 
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
//			labels[i] = lookup(label); // FIXME: replace "lookup(label)" with just "label" for actual training with 1000 classes
			labels[i] = label;
		} 

		JTensorFloatTuple pair = new JTensorFloatTuple(new JTensorFloat(images, dims), 
														new JTensorFloat(labels, new int[]{batch}));
		ArithStats.timing("sliceLmdbFloat", begin);
		return pair;
	}
	
//	// The lookup method is for testing only to force the class label to become 10 classes instead of 1000
//	static int[] map = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
//	static int lookup(int x) {
//		for(int i=0; i<10; i++) {
//			if(map[i] == x) {
//				return i;
//			}
//			else if (map[i] == -1) {
//				map[i] = x;
//				return i;
//			}
//		}
//		throw new RuntimeException("more than 10 categories");
//	}

	/**
	 * The method to set up Factory for stream loading from LMDB
	 *
	 * @param dims
	 * @param path the path of the LMDB data
	 * @param os the operating system of concern
	 * @return
	 */
	public static LmdbFactory getFactory(int[] dims, String path, LmdbUtils.OS os, int max_size) {
		return new LmdbFactory(dims, path, os, max_size);
	}
 
	public static LmdbFactory getFactory(String path, int max_size, int[] dims) { 
		return getFactory(dims, path, LmdbUtils.OS.LINUX, max_size);
	}
	 
	public String toString() { return "LMDB " + Arrays.toString(dims); }
}

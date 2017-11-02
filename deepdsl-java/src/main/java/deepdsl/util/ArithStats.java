package deepdsl.util;

import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Stream;

import deepdsl.cudnn.JCudaFunction;

/**
 * This class collects the number of calls for each arith op during program running
 *
 */
public class ArithStats {
	public static final double NANO_FOR_ONE_SEC = 1000000000.0;

	//The flag to turn on/off stats
	public static boolean isStats = false;

	public final static Map<String, Integer> opCounts = new HashMap<>();

	public final static Map<String, Long> timeMetricCounts = new HashMap<>();

	/**
	 * This method is called from every instrumented arith_op method to collect stat information
	 * with regards to tht # of calls for each arith_op and the time spent on each arith_op
	 *
	 * @param op
	 * @param time
	 */
	private static void doTime(String op, long time) {
		time = System.nanoTime() - time;
		Integer count = opCounts.get(op);
		opCounts.put(op, (count != null) ? count + 1 : 1); 
		Long value = timeMetricCounts.get(op);
		timeMetricCounts.put(op, (value != null) ? value + time : time);
	}

	public static void timing(String op, long time) {
		if (isStats) { 
			doTime(op, time);
		}
	}

	public static void cuda_timing(String op, long time) {
		if (isStats) {
			JCudaFunction.sync();
			doTime(op, time);
		}
	}

	/**
	 * This method outputs the stats for the total # of calls for each arith_op involved during program running;
	 * it also outputs
	 *
	 * @return void
	 */
	public static String outStats() {
		StringBuilder sb = new StringBuilder();
		
		if(isStats) { 
			sb.append("\n");
			sb.append("===========Stats===========\n");
			sb.append("arith_op | count\n");
			sb.append("---------------------------\n");
			for (Map.Entry<String, Integer> entry : sortByValue(opCounts).entrySet()) {
				sb.append(entry.getKey() + " | " + entry.getValue());
				sb.append("\n");
			}

			sb.append("---------------------------\n");
			sb.append("arith_op | time\n");
			sb.append("---------------------------\n");
			Long total = 0L;
			for (Map.Entry<String, Long> entry : sortByValue(timeMetricCounts).entrySet()) {
				total += entry.getValue();
				sb.append(entry.getKey() + " | " + entry.getValue() / NANO_FOR_ONE_SEC);
				sb.append("\n");
			}
			sb.append("---------------------------\n");
			sb.append("total | " + total / NANO_FOR_ONE_SEC + "\n");
			sb.append("===========================");
		}
		return sb.toString();
	}

	/**
	 * This method sorts a Map by its value in descending order by leveraging Java 8 stream and the new Comparator
	 *
	 * @param map The map to be sorted
	 * @return sorted Map
	 */
	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue( Map<K, V> map)
	{
		Map<K, V> result = new LinkedHashMap<>();
		Comparator<Map.Entry<K, V>> comparator = (entry1, entry2) -> entry1.getValue().compareTo(
				entry2.getValue());
		Stream<Map.Entry<K, V>> st = map.entrySet().stream();
		st.sorted(comparator.reversed()).forEachOrdered(e -> result.put(e.getKey(), e.getValue()));
		return result;
	}
}

package data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Utils {
	public static <T> List<T> makeSubset(List<T> superset, int size) {
		int trueSize = Math.min(superset.size(), size);
		List<T> subset = new ArrayList<T>(trueSize);
		Collections.shuffle(superset);
		for (int i = 0; i < trueSize; i++) {
			subset.add(superset.get(i));
		}
		return subset;
	}
	
	public static List<Instance<Integer, Boolean>> IntegerAttributeBooleanInstances(int size, int dimensions, int booleanFunction) {
		Random random = new Random();
		List<Instance<Integer, Boolean>> instances = new ArrayList<Instance<Integer, Boolean>>(size);
		for (int i = 0; i < size; i++) {
			List<Integer> nextAttributes = new ArrayList<Integer>();
			boolean res = random.nextBoolean();
			if (res) {
				nextAttributes.add(1);
			} else {
				nextAttributes.add(-1);
			}
			for (int j = 1; j < dimensions; j++) {
				boolean next = random.nextBoolean();
				
				switch(booleanFunction) {
				case 0:
					res &= next;
					break;
				case 1:
					res |= next;
					break;
				default:
					res ^= next;
				}
				
				if (next) {
					nextAttributes.add(1);
				} else {
					nextAttributes.add(-1);
				}
			}
			instances.add(new Instance<Integer, Boolean>(nextAttributes, res));
		}
		return instances;
	}
	
	public static List<Instance<Boolean, Boolean>> BooleanInstances(int size, int dimensions, int booleanFunction) {
		Random random = new Random();
		List<Instance<Boolean, Boolean>> instances = new ArrayList<Instance<Boolean, Boolean>>(size);
		for (int i = 0; i < size; i++) {
			List<Boolean> nextAttributes = new ArrayList<Boolean>();
			boolean res = random.nextBoolean();
			nextAttributes.add(res);
			for (int j = 1; j < dimensions; j++) {
				boolean next = random.nextBoolean();
				
				switch(booleanFunction) {
				case 0:
					res &= next;
					break;
				case 1:
					res |= next;
					break;
				default:
					res ^= next;
				}
				nextAttributes.add(next);
			}
			instances.add(new Instance<Boolean, Boolean>(nextAttributes, res));
		}
		return instances;
	}
}

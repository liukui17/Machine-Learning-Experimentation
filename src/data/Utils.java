package data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
}

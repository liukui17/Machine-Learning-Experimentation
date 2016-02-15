package data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
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
	
	public static List<Instance<Integer, Boolean>> IntegerAttributeMOfNInstances(int size, int m, int n, boolean atLeast) {
		Random random = new Random();
		List<Instance<Integer, Boolean>> instances = new ArrayList<Instance<Integer, Boolean>>(size);
		for (int i = 0; i < size; i++) {
			List<Integer> attributes = new ArrayList<Integer>(n);
			int numTrue = 0;
			for (int j = 0; j < n; j++) {
				boolean next = random.nextBoolean();
				if (next) {
					numTrue++;
					attributes.add(1);
				} else {
					attributes.add(-1);
				}
			}
			if (atLeast) {
				instances.add(new Instance<Integer, Boolean>(attributes, numTrue >= m));
			} else {
				instances.add(new Instance<Integer, Boolean>(attributes, numTrue == m));
			}
		}
		return instances;
	}
	
	public static List<Instance<Boolean, Boolean>> MOfNInstances(int size, int m, int n, boolean atLeast) {
		Random random = new Random();
		List<Instance<Boolean, Boolean>> instances = new ArrayList<Instance<Boolean, Boolean>>(size);
		for (int i = 0; i < size; i++) {
			List<Boolean> attributes = new ArrayList<Boolean>(n);
			int numTrue = 0;
			for (int j = 0; j < n; j++) {
				boolean next = random.nextBoolean();
				if (next) {
					numTrue++;
				}
				attributes.add(next);
			}
			if (atLeast) {
				instances.add(new Instance<Boolean, Boolean>(attributes, numTrue >= m));
			} else {
				instances.add(new Instance<Boolean, Boolean>(attributes, numTrue == m));
			}
		}
		return instances;
	}
	
	public static double intLnNorm(List<Integer> first, List<Integer> second, int n) {
		double res = 0.0;
		for (int i = 0; i < first.size(); i++) {
			res += Math.pow(Math.abs(first.get(i) - second.get(i)), n);
		}
		return res;
	}
	
	public static double intLInfinityNorm(List<Integer> first, List<Integer> second) {
		double res = 0.0;
		for (int i = 0; i < first.size(); i++) {
			double nextAttributeDistance = Math.abs(first.get(i) - second.get(i));
			if (nextAttributeDistance > res) {
				res = nextAttributeDistance;
			}
		}
		return res;
	}
	
	public static double doubleLnNorm(List<Double> first, List<Double> second, int n) {
		double res = 0.0;
		for (int i = 0; i < first.size(); i++) {
			res += Math.pow(Math.abs(first.get(i) - second.get(i)), n);
		}
		return res;
	}
	
	public static double doubleLInfinityNorm(List<Double> first, List<Double> second) {
		double res = 0.0;
		for (int i = 0; i < first.size(); i++) {
			double nextAttributeDistance = Math.abs(first.get(i) - second.get(i));
			if (nextAttributeDistance > res) {
				res = nextAttributeDistance;
			}
		}
		return res;
	}
	
	public static <L> L getMajority(Map<L, Integer> votesMap) {
		Iterator<L> it = votesMap.keySet().iterator();
		L best = null;
		if (it.hasNext()) {
			best = it.next();
			int bestScore = votesMap.get(best);
			while (it.hasNext()) {
				L next = it.next();
				int nextScore = votesMap.get(next);
				if (nextScore > bestScore) {
					best = next;
					bestScore = nextScore;
				}
			}
		}
		return best;
	}
	
	public static <L> L getHighestScorer(Map<L, Double> scoresMap) {
		Iterator<L> it = scoresMap.keySet().iterator();
		L best = null;
		if (it.hasNext()) {
			best = it.next();
			double bestScore = scoresMap.get(best);
			while (it.hasNext()) {
				L next = it.next();
				double nextScore = scoresMap.get(next);
				if (nextScore > bestScore) {
					best = next;
					bestScore = nextScore;
				}
			}
		}
		return best;
	}
	
	/**
	 * compute sigmoid of some input
	 */
	public static double sigmoid(double input) {
		return 1.0 / (1.0 + Math.exp(-input));
	}
	
	/**
	 * compute derivative of sigmoid at input
	 */
	public static double dSigmoid(double input) {
		double sig = sigmoid(input);
		return sig * (1.0 - sig);
	}
	
	/**
	 * compute derivative of hyperbolic tangent at input
	 */
	public static double dTanh(double input) {
		return 1 - Math.pow(Math.tanh(input), 2.0);
	}
	
	public static double rectifiedLinear(double input, double threshold) {
		return Math.max(threshold, input);
	}
	
	public static double dRectifiedLinear(double input, double threshold) {
		if (input > threshold) {
			return 1.0;
		} else {
			return 0.0;
		}
	}
}

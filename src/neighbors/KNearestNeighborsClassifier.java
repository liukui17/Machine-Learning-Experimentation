package neighbors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import data.Instance;
import data.NumericalInstance;

public class KNearestNeighborsClassifier<L> {
	int k;
	int n;
	Map<L, List<Instance<Integer, L>>> instances;
	
	public KNearestNeighborsClassifier(List<Instance<Integer, L>> instances, int k, int n) {
		this.k = k;
		this.n = n;
		this.instances = new HashMap<L,List<Instance<Integer, L>>>();
		for (Instance<Integer, L> instance : instances) {
			L nextLabel = instance.getLabel();
			if (this.instances.containsKey(nextLabel)) {
				this.instances.get(nextLabel).add(instance);
			} else {
				List<Instance<Integer, L>> subset = new LinkedList<Instance<Integer, L>>();
				subset.add(instance);
				this.instances.put(nextLabel, subset);
			}
		}
	}
	
/*	public double computeConditionalAttributeValue(int attributeIndex, Number thisValue, L label) {
		double numInstancesWithValue = 0.0;
		double numInstancesWithValueAndLabel = 0.0;
		Set<L> labelSet = instances.keySet();
		for (L nextLabel : labelSet) {
			List<Instance<Integer, L>> nextInstances = instances.get(nextLabel);
			for (Instance<Integer, L> instance : nextInstances) {
				if (instance.getAttributeValue(attributeIndex).equals(thisValue)) {
					numInstancesWithValue++;
					if (nextLabel.equals(label)) {
						numInstancesWithValueAndLabel++;
					}
				}
			}
		}
		return numInstancesWithValueAndLabel / numInstancesWithValue;
	}
	
	public double computeValueDifferenceMeasure(int attributeValue, Number thisValue, Number otherValue) {
		double res = 0.0;
		Set<L> labelSet = instances.keySet();
		for (L label : labelSet) {
			
		}
	} */
	
	public L predict(List<Integer> newData) {
		PriorityQueue<InstanceDistancePair> kNearest = getKNearest(newData);
		Map<L, Double> weightedVotes = new HashMap<L, Double>();
		for (InstanceDistancePair pair : kNearest) {
			L label = pair.instance.getLabel();
			if (weightedVotes.containsKey(label)) {
				weightedVotes.put(label, weightedVotes.get(label) + (1.0 / Math.pow(pair.dist, 2.0)));
			} else {
				weightedVotes.put(label, (1.0 / Math.pow(pair.dist, 2.0)));
			}
		}
		L predict = null;
		double weight = -Double.MAX_VALUE;
		for (L label : weightedVotes.keySet()) {
			double nextWeight = weightedVotes.get(label);
			if (nextWeight > weight) {
				weight = nextWeight;
				predict = label;
			}
		}
		return predict;
	}
	
	private PriorityQueue<InstanceDistancePair> getKNearest(List<Integer> newData) {
		PriorityQueue<InstanceDistancePair> nearestK = new PriorityQueue<InstanceDistancePair>(k);
		for (L label : instances.keySet()) {
			List<Instance<Integer, L>> nextInstances = instances.get(label);
			for (Instance<Integer, L> instance : nextInstances) {
				double dist = 0.0;
				if (n <= 0) {
					dist = LInfinityNorm(instance.getAttributeValues(), newData);
				} else {
					dist = LnNorm(instance.getAttributeValues(), newData, n);
				}
				if (nearestK.size() < k) {
					nearestK.offer(new InstanceDistancePair(instance, dist));
				} else if (dist < nearestK.peek().dist) {
					nearestK.remove();
					nearestK.offer(new InstanceDistancePair(instance, dist));
				}
			}
		}
		return nearestK;
	}
	
	public double LnNorm(List<Integer> first, List<Integer> second, int n) {
		double res = 0.0;
		for (int i = 0; i < first.size(); i++) {
			res += Math.abs(Math.pow(first.get(i) - second.get(i), n));
		}
		return Math.pow(res, 1.0 / n);
	}
	
	public double LInfinityNorm(List<Integer> first, List<Integer> second) {
		double res = 0.0;
		for (int i = 0; i < first.size(); i++) {
			double nextAttributeDistance = Math.abs(first.get(i) - second.get(i));
			if (nextAttributeDistance > res) {
				res = nextAttributeDistance;
			}
		}
		return res;
	}
	
	private class InstanceDistancePair implements Comparable<InstanceDistancePair> {
		
		Instance<Integer, L> instance;
		double dist;
		
		public InstanceDistancePair(Instance<Integer, L> instance, double dist) {
			this.instance = instance;
			this.dist = dist;
		}
		
		public int compareTo(InstanceDistancePair other) {
			if (this.dist > other.dist) {
				return -1;
			} else if (this.dist == other.dist) {
				return 0;
			} else {
				return 1;
			}
		}
	}
}

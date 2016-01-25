package neighbors;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import data.Instance;
import data.IntegerInstanceDistancePair;
import data.Utils;

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
	
	public KNearestNeighborsClassifier(Map<L, List<Instance<Integer, L>>> instances, int k, int n) {
		this.k = k;
		this.n = n;
		this.instances = instances;
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
		PriorityQueue<IntegerInstanceDistancePair<L>> kNearest = getKNearest(newData);
		Map<L, Double> weightedVotes = new HashMap<L, Double>();
		for (IntegerInstanceDistancePair<L> pair : kNearest) {
			L label = pair.getInstance().getLabel();
			if (weightedVotes.containsKey(label)) {
				weightedVotes.put(label, weightedVotes.get(label) + (1.0 / Math.pow(pair.getDistance(), 2.0)));
			} else {
				weightedVotes.put(label, (1.0 / Math.pow(pair.getDistance(), 2.0)));
			}
		}
		return Utils.getHighestScorer(weightedVotes);
	}
	
	private PriorityQueue<IntegerInstanceDistancePair<L>> getKNearest(List<Integer> newData) {
		PriorityQueue<IntegerInstanceDistancePair<L>> nearestK = new PriorityQueue<IntegerInstanceDistancePair<L>>(k);
		for (L label : instances.keySet()) {
			List<Instance<Integer, L>> nextInstances = instances.get(label);
			for (Instance<Integer, L> instance : nextInstances) {
				double dist = 0.0;
				if (n <= 0) {
					dist = Utils.intLInfinityNorm(instance.getAttributeValues(), newData);
				} else {
					dist = Utils.intLnNorm(instance.getAttributeValues(), newData, n);
				}
				if (nearestK.size() < k) {
					nearestK.offer(new IntegerInstanceDistancePair<L>(instance, dist));
				} else if (dist < nearestK.peek().getDistance()) {
					nearestK.remove();
					nearestK.offer(new IntegerInstanceDistancePair<L>(instance, dist));
				}
			}
		}
		return nearestK;
	}
}

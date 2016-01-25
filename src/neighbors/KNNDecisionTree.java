package neighbors;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import data.Chi;
import data.Instance;
import data.IntegerInstanceDistancePair;
import data.Utils;

public class KNNDecisionTree<L> {
	Map<L, List<Instance<Integer, L>>> trainingExamples;
	int dimensionality;
	DecisionNode root;
	int trainingDataSize;
	double significanceThreshold;
	int n, k;
	int consider;
	Random random;

	public KNNDecisionTree(List<Instance<Integer, L>> trainingExamples, int consider, double significanceThreshold, int n, int k) {
		this.dimensionality = trainingExamples.get(0).getDimensionality();
		this.trainingDataSize = trainingExamples.size();
		this.trainingExamples = new HashMap<L, List<Instance<Integer, L>>>();
		this.significanceThreshold = significanceThreshold;
		this.n = n;
		this.k = k;
		this.consider = consider;
		random = new Random();

		for (Instance<Integer, L> instance : trainingExamples) {
			L nextLabel = instance.getLabel();
			if (this.trainingExamples.containsKey(nextLabel)) {
				this.trainingExamples.get(nextLabel).add(instance);
			} else {
				List<Instance<Integer, L>> newSubSubset = new LinkedList<Instance<Integer, L>>();
				newSubSubset.add(instance);
				this.trainingExamples.put(nextLabel, newSubSubset);
			}
		}
		root = new DecisionNode(this.trainingExamples);
	}

	public L predict(List<Integer> newData) {
		DecisionNode decider = findDecidingNode(newData);
		return decider.knn.predict(newData);
	}

	private DecisionNode findDecidingNode(List<Integer> newData) {
		DecisionNode decider = root;
		while (decider != null && decider.children != null) {
			DecisionNode next = decider.children.get(newData.get(decider.splitAttribute));
			if (next == null) {
				return decider;
			}
			decider = next;
		}
		return decider;
	}

	public int getTrainingDataSize() {
		return trainingDataSize;
	}

	class DecisionNode {
		int trainingSubsetSize;
		int splitAttribute = -1;
		Map<Integer, DecisionNode> children = null;
		double entropy = 0.0;
		KNearestNeighborsClassifier<L> knn;

		public DecisionNode(Map<L, List<Instance<Integer, L>>> trainingSubset) {
			knn = new KNearestNeighborsClassifier<L>(trainingSubset, k, n);
			this.trainingSubsetSize = getSubsetSize(trainingSubset);
			if (trainingSubset != null && trainingSubset.size() != 0) {
				entropy = computeSubsetEntropy(trainingSubset);

				if (entropy != 0.0) {
					this.splitAttribute = getSplitAttribute(trainingSubset);
					if (this.splitAttribute != -1) {
						Map<Integer, Map<L, List<Instance<Integer, L>>>> partitionedSubsets = partition(
								this.splitAttribute, trainingSubset);
						if (isStatisticallySignificant(trainingSubset, partitionedSubsets)) {
							children = new HashMap<Integer, DecisionNode>(partitionedSubsets.size(), (float) 1.0);
							Iterator<Integer> iter = partitionedSubsets.keySet().iterator();
							while (iter.hasNext()) {
								Integer next = iter.next();
								Map<L, List<Instance<Integer, L>>> nextSubset = partitionedSubsets.get(next);
								children.put(next, new DecisionNode(nextSubset));
							}
						}
					}
				}
			}
		}

		public boolean isStatisticallySignificant(Map<L, List<Instance<Integer, L>>> trainingSubset,
				Map<Integer, Map<L, List<Instance<Integer, L>>>> subset) {
			double testStatistic = computeTestStatistic(trainingSubset, subset);
			double crit = Chi.critchi(significanceThreshold, (trainingSubset.size() - 1) * (subset.size() - 1));
			return testStatistic >= crit;
		}

		public double computeTestStatistic(Map<L, List<Instance<Integer, L>>> trainingSubset,
				Map<Integer, Map<L, List<Instance<Integer, L>>>> subset) {
			double res = 0.0;
			for (Integer value : subset.keySet()) {
				double numWithValue = 0.0;
				for (L label : subset.get(value).keySet()) {
					numWithValue += subset.get(value).get(label).size();
				}
				for (L label : trainingSubset.keySet()) {
					double ratio = (numWithValue * trainingSubset.get(label).size()) / trainingSubsetSize;
					List<Instance<Integer, L>> instancesWithValueAndLabel = subset.get(value).get(label);
					if (instancesWithValueAndLabel == null) {
						res += ratio;
					} else {
						res += Math.pow((instancesWithValueAndLabel.size() - ratio), 2.0) / ratio;
					}
				}
			}
			return res;
		}

		public int getSubsetSize(Map<L, List<Instance<Integer, L>>> map) {
			int total = 0;
			Collection<List<Instance<Integer, L>>> values = map.values();
			for (List<Instance<Integer, L>> list : values) {
				total += list.size();
			}
			return total;
		}

		public Map<Integer, Map<L, List<Instance<Integer, L>>>> partition(int index, Map<L, List<Instance<Integer, L>>> trainingSubset) {
			Map<Integer, Map<L, List<Instance<Integer, L>>>> partitionedSubsets = new HashMap<Integer, Map<L, List<Instance<Integer, L>>>>();
			Collection<List<Instance<Integer, L>>> values = trainingSubset.values();
			for (List<Instance<Integer, L>> nextLabelList : values) {
				for (Instance<Integer, L> nextInstance : nextLabelList) {
					Integer nextAttributeValue = nextInstance.getAttributeValue(index);
					L nextLabel = nextInstance.getLabel();
					if (partitionedSubsets.containsKey(nextAttributeValue)) {
						Map<L, List<Instance<Integer, L>>> subset = partitionedSubsets.get(nextAttributeValue);
						if (subset.containsKey(nextLabel)) {
							subset.get(nextLabel).add(nextInstance);
						} else {
							List<Instance<Integer, L>> newSubSubset = new LinkedList<Instance<Integer, L>>();
							newSubSubset.add(nextInstance);
							subset.put(nextLabel, newSubSubset);
						}
					} else {
						Map<L, List<Instance<Integer, L>>> newSubset = new HashMap<L, List<Instance<Integer, L>>>(
								trainingSubset.size(), (float) 1.0);
						List<Instance<Integer, L>> newSubSubset = new LinkedList<Instance<Integer, L>>();
						newSubSubset.add(nextInstance);
						newSubset.put(nextLabel, newSubSubset);
						partitionedSubsets.put(nextAttributeValue, newSubset);
					}
				}
			}
			return partitionedSubsets;
		}

		public double computeSubsetEntropy(Map<L, List<Instance<Integer, L>>> subset) {
			double entropy = 0.0;
			double total = getSubsetSize(subset);
			Set<L> labels = subset.keySet();
			for (L label : labels) {
				double nextLabelProbability = ((double) subset.get(label).size()) / total;
				entropy += (-nextLabelProbability) * Math.log(nextLabelProbability);
			}
			return entropy;
		}

		public double computeInformationGain(Map<Integer, Map<L, List<Instance<Integer, L>>>> subsetsMap) {
			double entropySum = 0.0;
			Collection<Map<L, List<Instance<Integer, L>>>> subsets = subsetsMap.values();
			for (Map<L, List<Instance<Integer, L>>> map : subsets) {
				entropySum += (computeSubsetEntropy(map) * getSubsetSize(map));
			}
			return entropy - (entropySum / trainingSubsetSize);
		}

		public int getSplitAttribute(Map<L, List<Instance<Integer, L>>> trainingSubset) {
		//	Set<Integer> consider = getConsideredAttributes();
			double currentGain = 0.0;
			int currentAttribute = -1;

		//	for (Integer i : consider) {
			for (int i = 0; i < dimensionality; i++) {
				Map<Integer, Map<L, List<Instance<Integer, L>>>> subsetsMap = partition(i, trainingSubset);

				double nextGain = computeInformationGain(subsetsMap);

				if (nextGain > currentGain) {
					currentGain = nextGain;
					currentAttribute = i;
				}
			}
		//	}
			return currentAttribute;
		}
		
		private Set<Integer> getConsideredAttributes() {
			Set<Integer> attributesToConsider = new HashSet<Integer>();
			for (int i = 0; i < consider; i++) {
				int next = random.nextInt(dimensionality);
				
				while (attributesToConsider.contains(next)) {
					next = random.nextInt(dimensionality);
				}
				attributesToConsider.add(next);
			}
			return attributesToConsider;
		}
	}
}
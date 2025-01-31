package tree;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import data.Instance;

/**
 * This is a subtype of the general DecisionTree. It uses an entropy-based
 * measure (such as information gain) to greedily decide the splitting attribute.
 */
public class EntropyDecisionTree<A, L> extends DecisionTree<A, L> {

	/** the root node of the decision tree */
	DecisionNode root;

	public EntropyDecisionTree(List<Instance<A, L>> trainingExamples, double significanceThreshold) {
		super(trainingExamples, significanceThreshold);
		root = new DecisionNode(this.trainingExamples, -1);
	}

	@Override
	DecisionTree<A, L>.DecisionNode findDecidingNode(List<A> newData) {
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

	class DecisionNode extends DecisionTree<A, L>.DecisionNode {

		/**
		 * a map from each distinct value (assumed by the attribute specified by
		 * this.splitIndex) to a decision node with the a training example
		 * subset in which all instances have that value (for the attribute
		 * specified by this.splitAttribute)
		 */
		Map<A, DecisionNode> children = null;

		public DecisionNode(Map<L, List<Instance<A, L>>> trainingSubset, int parentDepth) {
			super(trainingSubset, parentDepth);
			if (trainingSubset != null && trainingSubset.size() != 0) {
				entropy = computeSubsetEntropy(trainingSubset);
				
				prediction = findMajority(trainingSubset);

				/*
				 * There is no need to check every attribute, searching for the
				 * optimal attribute to split on, if there is no entropy in the
				 * current training instance subset to begin with; the data is
				 * already all of the same label.
				 */
				if (entropy != 0.0) {
					this.splitAttribute = getSplitAttribute(trainingSubset);
					if (this.splitAttribute != -1) {
						Map<A, Map<L, List<Instance<A, L>>>> partitionedSubsets = partition(this.splitAttribute,
								trainingSubset);
						if (isStatisticallySignificant(trainingSubset, partitionedSubsets)) {
							children = new HashMap<A, DecisionNode>(partitionedSubsets.size(), (float) 1.0);
							Iterator<A> iter = partitionedSubsets.keySet().iterator();
							while (iter.hasNext()) {
								A next = iter.next();
								Map<L, List<Instance<A, L>>> nextSubset = partitionedSubsets.get(next);
								children.put(next, new DecisionNode(nextSubset, depth));
							}
						}
					}
				}
				
				/*
				 * If children == null, then this node is a leaf so
				 * increment the leafCount. Further, since it is a
				 * leaf node, we've maximized the length of this path.
				 * Hence, we can consider updating the tree height
				 * using the depth at this node.
				 */
				if (children == null) {
					leafCount++;
					if (depth > height) {
						height = depth;
					}
				}
			}
		}

		/**
		 * Computes the information gained by splitting on a certain attribute.
		 * 
		 * @param subsetsMap
		 *            a partition of the training examples by the different
		 *            possible values of the attribute being split upon
		 * @return the information gain of the attribute
		 */
		public double computeInformationGain(Map<A, Map<L, List<Instance<A, L>>>> subsetsMap) {
			double entropySum = 0.0;
			Collection<Map<L, List<Instance<A, L>>>> subsets = subsetsMap.values();
			for (Map<L, List<Instance<A, L>>> map : subsets) {
				entropySum += (computeSubsetEntropy(map) * getSubsetSize(map));
			}
			return entropy - (entropySum / trainingSubsetSize);
		}

		/**
		 * Computes the split information due to splitting on a certain
		 * attribute.
		 * 
		 * @param subsetsMap
		 *            a partition of the training examples by the different
		 *            possible values of the attribute being split upon
		 * @return the split information of the attribute
		 */
		public double computeSplitInformation(Map<A, Map<L, List<Instance<A, L>>>> subsetsMap) {
			double splitInfo = 0.0;
			Collection<Map<L, List<Instance<A, L>>>> subsets = subsetsMap.values();
			for (Map<L, List<Instance<A, L>>> map : subsets) {
				double mapSize = getSubsetSize(map);
				splitInfo -= (mapSize / trainingSubsetSize) * Math.log(mapSize / trainingSubsetSize);
			}
			return splitInfo;
		}

		/**
		 * Computes the gain ratio due to splitting on a certain attribute,
		 * defined by the ratio of the information gain over the split
		 * information.
		 * 
		 * @param subsetsMap
		 *            a partition of the training examples by the different
		 *            possible values of the attribute being split upon
		 * @return the gain ratio of the attribute
		 */
		public double computeGainRatio(Map<A, Map<L, List<Instance<A, L>>>> subsetsMap) {
			double entropySum = 0.0;
			double splitInfo = 0.0;
			Collection<Map<L, List<Instance<A, L>>>> subsets = subsetsMap.values();
			for (Map<L, List<Instance<A, L>>> map : subsets) {
				double subsetSize = getSubsetSize(map);
				entropySum += (computeSubsetEntropy(map) * subsetSize);
				splitInfo -= (subsetSize / trainingSubsetSize) * Math.log(subsetSize / trainingSubsetSize);
			}
			if (splitInfo != 0.0) {
				return (entropy - (entropySum / trainingSubsetSize)) / splitInfo;
			} else {
				return 0.0;
			}
			
			/*
			 * Although this makes the code more readable, reduces redundancy and has
			 * better style in general, it is (much) less efficient because it reloops
			 * over the subsets of the map. While this shouldn't cost much on attributes
			 * with few values, we are still doing this second loop on every attribute
			 * at every node.
			 */
		/*	double gain = computeInformationGain(subsetsMap);
			double splitInfo = computeSplitInformation(subsetsMap);
			if (splitInfo == 0.0) {
				return 0.0;
			} else {
				return gain / splitInfo;
			} */
		}

		/**
		 * Finds the index of the optimal attribute to split on.
		 * 
		 * @return the index of the optimal attribute and -1 if there is no
		 *         longer any utility (defined below) in splitting further
		 */
		public int getSplitAttribute(Map<L, List<Instance<A, L>>> trainingSubset) {
			double currentGain = 0.0;
			int currentAttribute = -1;
			for (int i = 0; i < dimensionality; i++) {
				Map<A, Map<L, List<Instance<A, L>>>> subsetsMap = partition(i, trainingSubset);

				/*
				 * The optimal attribute is the one that yields the highest
				 * information gain. Another option that is useful for
				 * multi-valued attributes is to have the optimal attribute be
				 * the one that yields the highest gain ratio.
				 */
				double nextGain = computeInformationGain(subsetsMap);
			//	double nextGain = computeGainRatio(subsetsMap);

				if (nextGain > currentGain) {
					currentGain = nextGain;
					currentAttribute = i;
				}
			}
			return currentAttribute;
		}
	}
}

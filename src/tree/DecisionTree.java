package tree;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import data.Chi;
import data.Instance;

/**
 * A DecisionTree is a mutable object representing a decision tree learner. It
 * splits at each node by dividing the examples propagated to it into subsets,
 * each uniform in the value of a given attribute (the splitting attribute), which
 * is determined by some process (subclasses must implement this).
 * 
 * NOTE: the following only builds some of the necessary framework needed for
 * any decision tree learner but is certainly missing some necessary elements
 * such as the root node of the tree (since decision nodes may vary by implementation
 * of a decision tree learner, we leave this to the subclasses to implement); also,
 * we largely ignore design issues such as whether or not inheritance is appropriate
 * 
 * NOTE2: this is a classifier although it can probably be reasonably easily adapted to
 * do regression by thresholding values
 * 
 * TODO: implement a chi-squared split stopping criterion and/or post-pruning
 * to reduce overfitting (need statistical significance testing to give us
 * confidence that our "improvements" are legitimate improvements and not due to
 * noisy data)
 * 
 * @param <A>
 *            the type of the attributes (NOTE: currently assumes that all
 *            attributes have the same type)
 * @param <L>
 *            the type of the label
 */
public abstract class DecisionTree<A, L> {

	/**
	 * the training examples stored as a map from possible labels to lists of
	 * instances with those labels
	 */
	Map<L, List<Instance<A, L>>> trainingExamples;

	/** the dimensionality of the feature vectors */
	int dimensionality; // NOTE: currently assumes that all feature vectors have
						// same dimensionality

	/** the number of training examples used to train this decision tree */
	int trainingDataSize;
	
	/** the number of nodes in the decision tree */
	int nodeCount;
	
	/** 
	 * this holds the height of the decision tree, defined to be
	 * the longest path from the root to any of its leaves
	 */
	int height;
	
	/** the number of leaves in the decision tree */
	int leafCount;

	/**
	 * Constructs a new decision tree classifier.
	 * 
	 * @param trainingExamples
	 *            the training examples (pairs of feature vectors and labels)
	 *            used to build this tree
	 */
	public DecisionTree(List<Instance<A, L>> trainingExamples) {

		/*
		 * currently, we're assuming that all feature vectors have the same
		 * dimensionality
		 */
		this.dimensionality = trainingExamples.get(0).getDimensionality();
		this.trainingDataSize = trainingExamples.size();
		this.trainingExamples = new HashMap<L, List<Instance<A, L>>>();

		/*
		 * First, we perform some preprocessing by building up a map from each
		 * possible label to a corresponding list of instances that have that
		 * label.
		 */
		for (Instance<A, L> instance : trainingExamples) {
			L nextLabel = instance.getLabel();
			if (this.trainingExamples.containsKey(nextLabel)) {
				this.trainingExamples.get(nextLabel).add(instance);
			} else {
				List<Instance<A, L>> newSubSubset = new LinkedList<Instance<A, L>>();
				newSubSubset.add(instance);
				this.trainingExamples.put(nextLabel, newSubSubset);
			}
		}
		
		nodeCount = 0;
		height = 0;
		leafCount = 0;
	}

	/**
	 * Returns the most likely class for the given feature vector.
	 * 
	 * @param newData
	 *            the feature vector
	 * @return the highest probability label, i.e. the most common
	 */
	public L predict(List<A> newData) {
		DecisionNode decider = findDecidingNode(newData);
		return decider.prediction;
	}

	/**
	 * Propagates a new feature vector down the decision tree until a leaf node
	 * is hit, which will decide the class prediction.
	 * 
	 * @param newData
	 *            the feature vector
	 * @return the deciding node
	 */
	abstract DecisionNode findDecidingNode(List<A> newData);

	/**
	 * Gets the number of training examples used to train this tree.
	 * 
	 * @return the size of the training data set
	 */
	public int getTrainingDataSize() {
		return trainingDataSize;
	}
	
	/**
	 * Gets the number of nodes in the decision tree.
	 * 
	 * @return the number of nodes in the decision tree
	 */
	public int getNodeCount() {
		return nodeCount;
	}
	
	/**
	 * Gets the height of the decision tree.
	 * 
	 * @return the height of the decision tree
	 */
	public int getHeight() {
		return height;
	}
	
	/**
	 * Gets the number of leaves in the decision tree.
	 * 
	 * @return the number of leaves in the decision tree
	 */
	public int getLeafCount() {
		return leafCount;
	}
	
	/**
	 * Prints some statistics about this tree.
	 */
	public void printStats() {
		System.out.println("\tNode Count: " + nodeCount +
				   		   "\n\tLeaf Count: " + leafCount +
				   		   "\n\tHeight: " + height);
	}

	/**
	 * A DecisionNode is a mutable object representing a single node in a
	 * DecisionTree. As with a the containing decision tree class, this only
	 * sets up some necessary infrastructure for all decision nodes but also
	 * leaves out things such as pointers to children.
	 */
	abstract class DecisionNode {

		/** the total number of training instances in the subset */
		int trainingSubsetSize;

		/** the index of the attribute this node splits on */
		int splitAttribute = -1;

		/** the entropy of the training example subset at this node */
		double entropy = 0.0;

		/** the label that would be predicted at this node */
		L prediction = null;
		
		/** the depth of this node */
		int depth;
		
		/** 
		 * ideally, here would be a Map<A, DecisionNode> representing children
		 * nodes but since DecisionNodes must be subclassed, we can't place
		 * subtypes of DecisionNode in here
		 */

		/**
		 * Constructs a new DecisionNode. The DecisionTree is constructed
		 * recursively by constructing children DecisionNodes at construction of
		 * each DecisionNode.
		 * 
		 * NOTE: decision nodes do not store the training example subsets propagated
		 * to them since it is largely a waste of memory (all we care about is the
		 * classification, which can be computed at training time)
		 * 
		 * @param trainingSubset
		 *            the subset of training examples propagated down to this
		 *            node
		 */
		public DecisionNode(Map<L, List<Instance<A, L>>> trainingSubset, int parentDepth) {
			/*
			 * There's not a whole lot a DecisionNode can do in terms of
			 * construction since the other details are more specific to
			 * the implementation (such as how the splitting attribute is
			 * chosen and node types)
			 */
			this.trainingSubsetSize = getSubsetSize(trainingSubset);
			nodeCount++;
			depth = parentDepth + 1;
		}
		
		public boolean isStatisticallySignificant(Map<L, List<Instance<A, L>>> trainingSubset,
				Map<A, Map<L, List<Instance<A, L>>>> subset, double significanceThreshold) {
			double testStatistic = computeTestStatistic(trainingSubset, subset);
			double crit = Chi.critchi(significanceThreshold, (trainingSubset.size() - 1) * (subset.size() - 1));
			return testStatistic > crit;
		}
		
		public double computeTestStatistic(Map<L, List<Instance<A, L>>> trainingSubset,
				Map<A, Map<L, List<Instance<A, L>>>> subset) {
			double res = 0.0;
			for (A value : subset.keySet()) {
				double numWithValue = 0.0;
				for (L label : subset.get(value).keySet()) {
					numWithValue += subset.get(value).get(label).size();
				}
				for (L label : trainingSubset.keySet()) {
					double ratio = numWithValue * trainingSubset.get(label).size() / trainingSubsetSize;
					List<Instance<A, L>> instancesWithValueAndLabel = subset.get(value).get(label);
					if (instancesWithValueAndLabel == null) {
						res += res;
					} else {
						res += Math.pow((subset.get(value).get(label).size() - ratio), 2) / ratio;
					}
				}
			}
			return res;
		}

		/**
		 * Gets the total number of instances in the collection partitioned by
		 * label values.
		 * 
		 * @param map
		 *            a partition of instances by label values
		 * @return the total number of instances
		 */
		public int getSubsetSize(Map<L, List<Instance<A, L>>> map) {
			int total = 0;
			Collection<List<Instance<A, L>>> values = map.values();
			for (List<Instance<A, L>> list : values) {
				total += list.size();
			}
			return total;
		}

		/**
		 * Partitions the current collection of instances based on the possible
		 * values a given attribute (specified by the index) can assume.
		 * 
		 * @param index
		 *            the index of the attribute
		 * @return a partition of the current collection of training examples
		 */
		public Map<A, Map<L, List<Instance<A, L>>>> partition(int index, Map<L, List<Instance<A, L>>> trainingSubset) {
			Map<A, Map<L, List<Instance<A, L>>>> partitionedSubsets = new HashMap<A, Map<L, List<Instance<A, L>>>>();
			Collection<List<Instance<A, L>>> values = trainingSubset.values();
			for (List<Instance<A, L>> nextLabelList : values) {
				for (Instance<A, L> nextInstance : nextLabelList) {
					A nextAttributeValue = nextInstance.getAttributeValue(index);
					L nextLabel = nextInstance.getLabel();
					if (partitionedSubsets.containsKey(nextAttributeValue)) {
						Map<L, List<Instance<A, L>>> subset = partitionedSubsets.get(nextAttributeValue);
						if (subset.containsKey(nextLabel)) {
							subset.get(nextLabel).add(nextInstance);
						} else {
							List<Instance<A, L>> newSubSubset = new LinkedList<Instance<A, L>>();
							newSubSubset.add(nextInstance);
							subset.put(nextLabel, newSubSubset);
						}
					} else {
						Map<L, List<Instance<A, L>>> newSubset = new HashMap<L, List<Instance<A, L>>>(
								trainingSubset.size(), (float) 1.0);
						List<Instance<A, L>> newSubSubset = new LinkedList<Instance<A, L>>();
						newSubSubset.add(nextInstance);
						newSubset.put(nextLabel, newSubSubset);
						partitionedSubsets.put(nextAttributeValue, newSubset);
					}
				}
			}
			return partitionedSubsets;
		}

		/**
		 * Computes the entropy of a given subset of training examples.
		 * 
		 * @param subset
		 *            the subset of the training examples to compute the entropy
		 *            of
		 * @return the entropy of the training example subset
		 */
		public double computeSubsetEntropy(Map<L, List<Instance<A, L>>> subset) {
			double entropy = 0.0;
			double total = getSubsetSize(subset);
			Set<L> labels = subset.keySet();
			for (L label : labels) {
				double nextLabelProbability = ((double) subset.get(label).size()) / total;
				entropy += (-nextLabelProbability) * Math.log(nextLabelProbability);
			}
			return entropy;
		}

		/**
		 * Gets the most common label amongst the training examples at the
		 * current decision tree node.
		 * 
		 * @return the most common label
		 */
		public L findMajority(Map<L, List<Instance<A, L>>> trainingSubset) {
			L majority = null;
			int majorityCount = -1;
			Set<L> labelSet = trainingSubset.keySet();
			for (L label : labelSet) {
				int nextLabelCount = trainingSubset.get(label).size();
				if (nextLabelCount > majorityCount) {
					majority = label;
					majorityCount = nextLabelCount;
				}
			}
			return majority;
		}
		
		/**
		 * Gets the depth of this node.
		 * 
		 * @return the depth of this node
		 */
		public int getDepth() {
			return depth;
		}
	}
}
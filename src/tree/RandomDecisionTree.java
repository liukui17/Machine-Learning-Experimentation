package tree;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import data.Instance;

/**
 * This is a subclass of the general decision tree. It decides the
 * splitting attribute by random selection, splitting until there
 * is no longer any utility in splitting (for now, this is just
 * when the propagated subsets becomes pure in their labels).
 */
public class RandomDecisionTree<A, L> extends DecisionTree<A, L> {

	/** the root node of the decision tree */
	DecisionNode root;

	/** the random number generator used for deciding the splitting attribute */
	Random random;
	
	double significanceThreshold = 0.5;

	public RandomDecisionTree(List<Instance<A, L>> trainingExamples) {
		super(trainingExamples);
		random = new Random();
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
					this.splitAttribute = random.nextInt(dimensionality);
					Map<A, Map<L, List<Instance<A, L>>>> partitionedSubsets = partition(this.splitAttribute,
							trainingSubset);
					if (isStatisticallySignificant(trainingSubset, partitionedSubsets, significanceThreshold)) {
						children = new HashMap<A, DecisionNode>(partitionedSubsets.size(), (float) 1.0);
						Iterator<A> iter = partitionedSubsets.keySet().iterator();
						while (iter.hasNext()) {
							A next = iter.next();
							Map<L, List<Instance<A, L>>> nextSubset = partitionedSubsets.get(next);
							children.put(next, new DecisionNode(nextSubset, depth));
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
	}
}

package tree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import data.Instance;
import data.Utils;

/**
 * A Forest is a mutable object representing a decision tree ensemble learner,
 * an ensemble machine learning method built upon decision tree learners.
 * It performs classification based on majority vote of a committee of decision
 * trees
 * 
 * Bagging: The way bagging works is give each decision tree learner a randomized
 * subset of the training data. It reduces the tendency to overfit since it is
 * unlikely every decision tree in the committee overfits in exactly the same
 * manner. They are each trained on at least somewhat different data after all.
 * Intuitively, this improves upon decision trees when the committee members
 * already achieve decent accuracy (democracy!). Although, it is conceivable
 * this might fail when the committee trees don't achieve reasonably good
 * classification accuracies since there is low confidence in each vote.
 * This hasn't been tested but perhaps soon.
 * 
 * NOTE: the following only builds some of the necessary framework needed for
 * any forest learner but is certainly missing some necessary elements such as the
 * committee members (since decision trees may vary by implementation of a decision
 * tree learner, we leave this to the subclasses to implement); also, we largely
 * ignore design issues such as whether or not inheritance is appropriate
 * 
 * NOTE2: this is a classifier although it can probably be reasonably easily adapted to
 * do regression by thresholding values
 *
 * @param <A>
 *            the type of the attributes (NOTE: currently assumes that all
 *            attributes have the same type)
 * @param <L>
 *            the type of the label
 */
public abstract class Forest<A, L> {
	
	/** a list holding the committee of decision trees */
	List<DecisionTree<A, L>> committee;
	
	/** (variable names should make their purposes clear) */
	int minNodeCount = Integer.MAX_VALUE;
	int avgNodeCount;
	int maxNodeCount = Integer.MIN_VALUE;
	
	int minHeight = Integer.MAX_VALUE;
	int avgHeight;
	int maxHeight = Integer.MIN_VALUE;
	
	int minLeafCount = Integer.MAX_VALUE;
	int avgLeafCount;
	int maxLeafCount = Integer.MIN_VALUE;
	
	int totalNodeCount;

	public Forest(List<Instance<A, L>> trainingExamples, int subsetSize, int committeeSize) {
		committee = new ArrayList<DecisionTree<A, L>>(committeeSize);
	}

	public Forest(List<List<Instance<A, L>>> trainingSets) {
		committee = new ArrayList<DecisionTree<A, L>>(trainingSets.size());
	}

	public int getCommitteeSize() {
		return committee.size();
	}

	public L predict(List<A> newData) {
		Map<L, Double> counts = new HashMap<L, Double>();
		for (int i = 0; i < committee.size(); i++) {
			L newPrediction = committee.get(i).predict(newData);
			if (counts.containsKey(newPrediction)) {
				counts.put(newPrediction, counts.get(newPrediction) + 1.0);
			} else {
				counts.put(newPrediction, 1.0);
			}
		}
		return Utils.getHighestScorer(counts);
	}
	
	public int getMinNodeCount() { return minNodeCount; }
	public int getAvgNodeCount() { return avgNodeCount; }
	public int getMaxNodeCount() { return maxNodeCount; }
	
	public int getMinHeight() { return minHeight; }
	public int getAvgHeight() { return avgHeight; }
	public int getMaxHeight() { return maxHeight; }
	
	public int getMinLeafCount() { return minLeafCount; }
	public int getAvgLeafCount() { return avgLeafCount; }
	public int getMaxLeafCount() { return maxLeafCount; }
	
	public void printStats() {
		System.out.println("===== Node Counts =====");
		System.out.println("\tSmallest: " + minNodeCount +
						   "\n\tAverage: " + avgNodeCount +
						   "\n\tLargest: " + maxNodeCount + 
						   "\n\tTotal: " + totalNodeCount);
		System.out.println("===== Heights =====");
		System.out.println("\tShortest: " + minHeight +
						   "\n\tAverage: " + avgHeight +
						   "\n\tTallest: " + maxHeight);
		System.out.println("===== Leaf Counts =====");
		System.out.println("\tLeast: " + minLeafCount +
						   "\n\tAverage: " + avgLeafCount +
						   "\n\tMost: " + maxLeafCount);
	}
	
	void updateStats(DecisionTree<A, L> tree) {
		minNodeCount = Math.min(minNodeCount, tree.getNodeCount());
		maxNodeCount = Math.max(maxNodeCount, tree.getNodeCount());
		
		minHeight = Math.min(minHeight, tree.getHeight());
		maxHeight = Math.max(maxHeight, tree.getHeight());
		
		minLeafCount = Math.min(minLeafCount, tree.getLeafCount());
		maxLeafCount = Math.max(maxLeafCount, tree.getLeafCount());
		
		totalNodeCount += tree.getNodeCount();
	}
	
	void updateAverages(int totalHeights, int totalLeaves) {
		avgNodeCount = totalNodeCount / committee.size();
		avgHeight = totalHeights / committee.size();
		avgLeafCount = totalLeaves / committee.size();
	}
}

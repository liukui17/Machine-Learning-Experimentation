package tree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import data.Instance;

/**
 * A RandomForest is a mutable object representing a random forest learner,
 * an ensemble machine learning method built upon decision tree learners.
 * It uses a version of Bootstrap AGGregatING (short: "bagging") to construct
 * a committee of decision tree learners and performs classification based
 * on majority vote.
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
 * any random forest learner but is certainly missing some necessary elements
 * such as the committee members (since decision trees may vary by implementation
 * of a decision tree learner, we leave this to the subclasses to implement); also,
 * we largely ignore design issues such as whether or not inheritance is appropriate
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
public abstract class RandomForest<A, L> {
	
	/** a list holding the committee of decision trees */
	List<DecisionTree<A, L>> committee;
	
	/** (variable names should make their purposes clear) */
	int minNodeCount;
	int avgNodeCount;
	int maxNodeCount;
	
	int minHeight;
	int avgHeight;
	int maxHeight;
	
	int minLeafCount;
	int avgLeafCount;
	int maxLeafCount;

	public RandomForest(List<Instance<A, L>> trainingExamples, int subsetSize, int committeeSize) {
		committee = new ArrayList<DecisionTree<A, L>>(committeeSize);
	}

	public RandomForest(List<List<Instance<A, L>>> trainingSets) {
		committee = new ArrayList<DecisionTree<A, L>>(trainingSets.size());
	}

	public int getCommitteeSize() {
		return committee.size();
	}

	public L predict(List<A> newData) {
		Map<L, Integer> counts = new HashMap<L, Integer>();
		for (int i = 0; i < committee.size(); i++) {
			L newPrediction = committee.get(i).predict(newData);
			if (counts.containsKey(newPrediction)) {
				counts.put(newPrediction, counts.get(newPrediction) + 1);
			} else {
				counts.put(newPrediction, 1);
			}
		}
		L mode = null;
		int count = Integer.MIN_VALUE;
		Set<L> predictions = counts.keySet();
		for (L prediction : predictions) {
			int nextCount = counts.get(prediction);
			if (nextCount > count) {
				count = nextCount;
				mode = prediction;
			}
		}
		return mode;
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
						   "\n\tLargest: " + maxNodeCount);
		System.out.println("===== Heights =====");
		System.out.println("\tShortest: " + minHeight +
						   "\n\tAverage: " + avgHeight +
						   "\n\tTallest: " + maxHeight);
		System.out.println("==== Leaf Counts =====");
		System.out.println("\tLeast: " + minLeafCount +
						   "\n\tAverage: " + avgLeafCount +
						   "\n\tMost: " + maxLeafCount);
	}
}

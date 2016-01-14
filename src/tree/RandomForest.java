package tree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import data.Instance;

public abstract class RandomForest<A, L> {
	
	List<DecisionTree<A, L>> committee;
	
	/** (variable names make their purposes clear) */
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

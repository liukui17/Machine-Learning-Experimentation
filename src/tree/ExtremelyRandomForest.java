package tree;

import java.util.List;

import data.Instance;
import data.Utils;

/**
 * This is a subclass of Forest. It builds a forest of RandomDecisionTrees.
 */
public class ExtremelyRandomForest<A, L> extends Forest<A, L> {

	public ExtremelyRandomForest(List<Instance<A, L>> trainingExamples, double significanceThreshold, int subsetSize, int committeeSize) {
		super(trainingExamples, subsetSize, committeeSize);
		int totalHeights = 0;
		int totalLeaves = 0;
		for (int i = 0; i < committeeSize; i++) {
			DecisionTree<A, L> nextTree = new RandomDecisionTree<A, L>(Utils.makeSubset(trainingExamples, subsetSize), significanceThreshold);
			committee.add(nextTree);

			updateStats(nextTree);

			totalHeights += nextTree.getHeight();
			totalLeaves += nextTree.getLeafCount();
		}
		
		updateAverages(totalHeights, totalLeaves);
	}
	
	public ExtremelyRandomForest(List<List<Instance<A, L>>> trainingSets, double significanceThreshold) {
		super(trainingSets);
		int totalHeights = 0;
		int totalLeaves = 0;
		for (int i = 0; i < trainingSets.size(); i++) {
			DecisionTree<A, L> nextTree = new RandomDecisionTree<A, L>(trainingSets.get(i), significanceThreshold);
			committee.add(nextTree);

			updateStats(nextTree);

			totalHeights += nextTree.getHeight();
			totalLeaves += nextTree.getLeafCount();
		}
		
		updateAverages(totalHeights, totalLeaves);
	}

	public ExtremelyRandomForest(List<Instance<A, L>> trainingExamples, double significanceThreshold, int committeeSize) {
		this(trainingExamples, significanceThreshold, trainingExamples.size(), committeeSize);
	}
}

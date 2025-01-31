package tree;

import java.util.List;

import data.Instance;
import data.Utils;

public class BaggedTrees<A, L> extends Forest<A, L> {
	
	public static final int DEFAULT_SUBSET_RATIO = 2;

	public BaggedTrees(List<Instance<A, L>> trainingExamples, double significanceThreshold, int subsetSize, int committeeSize) {
		super(trainingExamples, subsetSize, committeeSize);
		int totalHeights = 0;
		int totalLeaves = 0;
		for (int i = 0; i < committeeSize; i++) {
			DecisionTree<A, L> nextTree = new EntropyDecisionTree<A, L>(Utils.makeSubset(trainingExamples, subsetSize), significanceThreshold);
			committee.add(nextTree);

			updateStats(nextTree);

			totalHeights += nextTree.getHeight();
			totalLeaves += nextTree.getLeafCount();
		}
		
		updateAverages(totalHeights, totalLeaves);
	}

	public BaggedTrees(List<List<Instance<A, L>>> trainingSets, double significanceThreshold) {
		super(trainingSets);
		int totalHeights = 0;
		int totalLeaves = 0;
		for (int i = 0; i < trainingSets.size(); i++) {
			DecisionTree<A, L> nextTree = new EntropyDecisionTree<A, L>(trainingSets.get(i), significanceThreshold);
			committee.add(nextTree);

			updateStats(nextTree);

			totalHeights += nextTree.getHeight();
			totalLeaves += nextTree.getLeafCount();
		}
		
		updateAverages(totalHeights, totalLeaves);
	}
	
	public BaggedTrees(List<Instance<A, L>> trainingExamples, double significanceThreshold, int committeeSize) {
		this(trainingExamples, significanceThreshold, trainingExamples.size() / DEFAULT_SUBSET_RATIO, committeeSize);
	}
}

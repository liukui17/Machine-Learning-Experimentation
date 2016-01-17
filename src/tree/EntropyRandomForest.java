package tree;

import java.util.List;

import data.Instance;
import data.Utils;

public class EntropyRandomForest<A, L> extends RandomForest<A, L> {
	
	public static final int DEFAULT_SUBSET_RATIO = 2;

	public EntropyRandomForest(List<Instance<A, L>> trainingExamples, int subsetSize, int committeeSize) {
		super(trainingExamples, subsetSize, committeeSize);
		int totalHeights = 0;
		int totalLeaves = 0;
		for (int i = 0; i < committeeSize; i++) {
			DecisionTree<A, L> nextTree = new EntropyDecisionTree<A, L>(Utils.makeSubset(trainingExamples, subsetSize));
			committee.add(nextTree);

			updateStats(nextTree);

			totalHeights += nextTree.getHeight();
			totalLeaves += nextTree.getLeafCount();
		}
		
		updateAverages(totalHeights, totalLeaves);
	}

	public EntropyRandomForest(List<List<Instance<A, L>>> trainingSets) {
		super(trainingSets);
		int totalHeights = 0;
		int totalLeaves = 0;
		for (int i = 0; i < trainingSets.size(); i++) {
			DecisionTree<A, L> nextTree = new EntropyDecisionTree<A, L>(trainingSets.get(i));
			committee.add(nextTree);

			updateStats(nextTree);

			totalHeights += nextTree.getHeight();
			totalLeaves += nextTree.getLeafCount();
		}
		
		updateAverages(totalHeights, totalLeaves);
	}
	
	public EntropyRandomForest(List<Instance<A, L>> trainingExamples, int committeeSize) {
		this(trainingExamples, trainingExamples.size() / DEFAULT_SUBSET_RATIO, committeeSize);
	}
}

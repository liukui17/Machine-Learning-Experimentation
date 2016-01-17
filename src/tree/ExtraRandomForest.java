package tree;

import java.util.List;

import data.Instance;
import data.Utils;

public class ExtraRandomForest<A, L> extends RandomForest<A, L> {
	
	public static final int DEFAULT_NOMINATION_RATIO = 25;
	
	public static final int DEFAULT_SUBSET_RATIO = 2;
	
	public ExtraRandomForest(List<Instance<A, L>> trainingExamples, int subsetSize, int committeeSize) {
		super(trainingExamples, subsetSize, committeeSize);
		int totalHeights = 0;
		int totalLeaves = 0;
		for (int i = 0; i < committeeSize; i++) {
			/*
			 * Default to nominating only 1/10 the number of total attributes at each iteration.
			 */
			DecisionTree<A, L> nextTree = new RandomEntropyDecisionTree<A, L>(Utils.makeSubset(trainingExamples, subsetSize),
					trainingExamples.get(0).getDimensionality() / DEFAULT_NOMINATION_RATIO);
			committee.add(nextTree);

			updateStats(nextTree);

			totalHeights += nextTree.getHeight();
			totalLeaves += nextTree.getLeafCount();
		}
		
		updateAverages(totalHeights, totalLeaves);
	}
	
	public ExtraRandomForest(List<Instance<A, L>> trainingExamples, int subsetSize, List<Integer> nominationCounts) {
		super(trainingExamples, subsetSize, nominationCounts.size());
		int totalHeights = 0;
		int totalLeaves = 0;
		for (int i = 0; i < nominationCounts.size(); i++) {
			DecisionTree<A, L> nextTree = new RandomEntropyDecisionTree<A, L>(Utils.makeSubset(trainingExamples, subsetSize),
					nominationCounts.get(i));
			committee.add(nextTree);
			
			updateStats(nextTree);
			
			totalHeights += nextTree.getHeight();
			totalLeaves += nextTree.getLeafCount();
		}
		
		updateAverages(totalHeights, totalLeaves);
	}

	public ExtraRandomForest(List<Instance<A, L>> trainingExamples, int committeeSize) {
		this(trainingExamples, trainingExamples.size() / DEFAULT_SUBSET_RATIO, committeeSize);
	}
}

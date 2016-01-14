package tree;

import java.util.List;

import data.Instance;
import data.Utils;

/**
 * This is a subclass of RandomForest. It builds a forest of RandomDecisionTrees.
 */
public class ExtremelyRandomForest<A, L> extends RandomForest<A, L> {

	public ExtremelyRandomForest(List<Instance<A, L>> trainingExamples, int subsetSize, int committeeSize) {
		super(trainingExamples, subsetSize, committeeSize);
		int totalHeights = 0;
		int totalLeaves = 0;
		for (int i = 0; i < committeeSize; i++) {
			DecisionTree<A, L> nextTree = new RandomDecisionTree<A, L>(Utils.makeSubset(trainingExamples, subsetSize));
			committee.add(nextTree);

			updateStats(nextTree);

			totalHeights += nextTree.getHeight();
			totalLeaves += nextTree.getLeafCount();
		}
		
		updateAverages(totalHeights, totalLeaves);
	}
	
	public ExtremelyRandomForest(List<List<Instance<A, L>>> trainingSets) {
		super(trainingSets);
		int totalHeights = 0;
		int totalLeaves = 0;
		for (int i = 0; i < trainingSets.size(); i++) {
			DecisionTree<A, L> nextTree = new RandomDecisionTree<A, L>(trainingSets.get(i));
			committee.add(nextTree);

			updateStats(nextTree);

			totalHeights += nextTree.getHeight();
			totalLeaves += nextTree.getLeafCount();
		}
		
		updateAverages(totalHeights, totalLeaves);
	}

	public ExtremelyRandomForest(List<Instance<A, L>> trainingExamples, int committeeSize) {
		this(trainingExamples, trainingExamples.size(), committeeSize);
	}
}

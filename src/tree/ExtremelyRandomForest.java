package tree;

import java.util.List;

import data.Instance;
import data.Utils;

public class ExtremelyRandomForest<A, L> extends RandomForest<A, L> {

	public ExtremelyRandomForest(List<Instance<A, L>> trainingExamples, int subsetSize, int committeeSize) {
		super(trainingExamples, subsetSize, committeeSize);
		for (int i = 0; i < committeeSize; i++) {
			committee.add(new RandomDecisionTree<A, L>(Utils.makeSubset(trainingExamples, subsetSize)));
		}
	}

	public ExtremelyRandomForest(List<Instance<A, L>> trainingExamples, int committeeSize) {
		this(trainingExamples, trainingExamples.size(), committeeSize);
	}

	public ExtremelyRandomForest(List<List<Instance<A, L>>> trainingSets) {
		super(trainingSets);
		for (int i = 0; i < trainingSets.size(); i++) {
			committee.add(new RandomDecisionTree<A, L>(trainingSets.get(i)));
		}
	}
}

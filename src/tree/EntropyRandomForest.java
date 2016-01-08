package tree;

import java.util.List;

import data.Instance;
import data.Utils;

public class EntropyRandomForest<A, L> extends RandomForest<A, L> {

	public EntropyRandomForest(List<Instance<A, L>> trainingExamples, int subsetSize, int committeeSize) {
		super(trainingExamples, subsetSize, committeeSize);
		for (int i = 0; i < committeeSize; i++) {
			committee.add(new EntropyDecisionTree<A, L>(Utils.makeSubset(trainingExamples, subsetSize)));
		}
	}

	public EntropyRandomForest(List<List<Instance<A, L>>> trainingSets) {
		super(trainingSets);
		for (int i = 0; i < trainingSets.size(); i++) {
			committee.add(new EntropyDecisionTree<A, L>(trainingSets.get(i)));
		}
	}
}

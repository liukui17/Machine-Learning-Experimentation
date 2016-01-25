package neighbors;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import data.Instance;
import data.Utils;

public class KNNRandomForest<L> {

	public static final int DEFAULT_NOMINATION_RATIO = 25;
	
	List<KNNDecisionTree<L>> committee;
	double significanceThreshold;
	int n, k;
	
	public KNNRandomForest(List<Instance<Integer, L>> trainingExamples, int committeeSize, double significanceThreshold, int n, int k) {
		this.significanceThreshold = significanceThreshold;
		this.n = n;
		this.k = k;
		committee = new ArrayList<KNNDecisionTree<L>>(committeeSize);
		for (int i = 0; i < committeeSize; i++) {
			committee.add(new KNNDecisionTree<L>(Utils.makeSubset(trainingExamples, 4 * trainingExamples.size() / 6),
					trainingExamples.get(0).getDimensionality() / DEFAULT_NOMINATION_RATIO, significanceThreshold, n, k));
		}
	}
	
	public L predict(List<Integer> newData) {
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
}

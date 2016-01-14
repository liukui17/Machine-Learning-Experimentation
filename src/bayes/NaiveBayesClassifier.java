package bayes;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import data.Instance;

/**
 * TODO: currently, this is very space inefficient, even though
 * it is pretty quick when predicting; optimize more!
 */
public class NaiveBayesClassifier<A, L> {

	Map<L, List<Map<A, List<Instance<A, L>>>>> trainingExamples;
	
	Map<L, Integer> labelCounts;
	
	int trainingSize;
	
	int dimensionality;
	
	public NaiveBayesClassifier(List<Instance<A, L>> trainingExamples) {
		this.trainingSize = trainingExamples.size();
		this.dimensionality = trainingExamples.get(0).getDimensionality();
		this.trainingExamples = train(trainingExamples);
		
		labelCounts = new HashMap<L, Integer>();
		for (Instance<A, L> instance : trainingExamples) {
			L nextLabel = instance.getLabel();
			if (labelCounts.containsKey(nextLabel)) {
				labelCounts.put(nextLabel, labelCounts.get(nextLabel) + 1);
			} else {
				labelCounts.put(nextLabel, 1);
			}
		}	
	}
	
	private Map<L, List<Map<A, List<Instance<A, L>>>>> train(List<Instance<A, L>> examples) {
		Map<L, List<Map<A, List<Instance<A, L>>>>> training = new HashMap<L, List<Map<A, List<Instance<A, L>>>>>();
		for (Instance<A, L> example : examples) {
			L nextLabel = example.getLabel();
			List<Map<A, List<Instance<A, L>>>> labelSet = training.get(nextLabel);
			if (labelSet == null) {
				List<Map<A, List<Instance<A, L>>>> newLabelSet = new ArrayList<Map<A, List<Instance<A, L>>>>(dimensionality);
				for (int i = 0; i < dimensionality; i++) {
					Map<A, List<Instance<A, L>>> attributeSet = new HashMap<A, List<Instance<A, L>>>();
					A nextAttributeValue = example.getAttributeValue(i);
					List<Instance<A, L>> instances = new LinkedList<Instance<A, L>>();
					instances.add(example);
					attributeSet.put(nextAttributeValue, instances);
					newLabelSet.add(attributeSet);
				}
				training.put(nextLabel, newLabelSet);
			} else {
				for (int i = 0; i < dimensionality; i++) {
					A nextAttributeValue = example.getAttributeValue(i);
					List<Instance<A, L>> instances = labelSet.get(i).get(nextAttributeValue);
					if (instances != null) {
						instances.add(example);
					} else {
						List<Instance<A, L>> newInstances = new LinkedList<Instance<A, L>>();
						newInstances.add(example);
						labelSet.get(i).put(nextAttributeValue, newInstances);
					}
				}
			}
		}
		return training;
	}
	
	public L predict(List<A> newData) {
		L prediction = null;
		double probability = -Double.MAX_VALUE;
		for (L label : trainingExamples.keySet()) {
			double logProbability = Math.log(((double) labelCounts.get(label)) / trainingSize);
			List<Map<A, List<Instance<A, L>>>> nextLabelSet = trainingExamples.get(label);
			for (int i = 0; i < newData.size(); i++) {
				Map<A, List<Instance<A, L>>> nextAttributeSet = nextLabelSet.get(i);
				double total = 0.0;
				for (A value : nextAttributeSet.keySet()) {
					total += nextAttributeSet.get(value).size();
				}
				double numerator = 1;
				List<Instance<A, L>> similar = nextAttributeSet.get(newData.get(i));
				if (similar != null) {
					numerator = similar.size();
				}
				logProbability += Math.log(numerator / total);
			}
			if (logProbability > probability) {
				prediction = label;
				probability = logProbability;
			}
		}
		return prediction;
	}
}

package neural;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import data.Instance;
import data.Utils;

public class FeedForwardNN<L> {
	
	double learningRate = 0.002;
	
	List<L> indexToLabel;
	
	List<List<NonLinearUnit>> layers;
	
	List<Instance<Integer, L>> trainingExamples;
	
	int dimensionality;
	
	public FeedForwardNN(List<Instance<Integer, L>> trainingExamples, List<L> indexToLabel, int[] layerSizes) {
		this.trainingExamples = trainingExamples;
		dimensionality = trainingExamples.get(0).getDimensionality();
		this.indexToLabel = indexToLabel;
		
		layers = new ArrayList<List<NonLinearUnit>>(layerSizes.length);
		List<NonLinearUnit> inputLayer = new ArrayList<NonLinearUnit>(layerSizes[0]);
		for (int i = 0; i < layerSizes[0]; i++) {
			inputLayer.add(new SigmoidUnit(dimensionality));
		//	inputLayer.add(new TanhUnit(dimensionality));
		//	inputLayer.add(new RectifiedLinearUnit(dimensionality));
		}
		for (int i = 1; i < layerSizes.length; i++) {
			List<NonLinearUnit> nextLayer = new ArrayList<NonLinearUnit>(layerSizes[i]);
			for (int j = 0; j < layerSizes[i]; j++) {
				nextLayer.add(new SigmoidUnit(layers.get(i - 1).size()));
			//	nextLayer.add(new TanhUnit(layers.get(i - 1).size()));
			//	nextLayer.add(new RectifiedLinearUnit(layers.get(i - 1).size()));
			}
			layers.add(nextLayer);
		}
	}
	
	public L predict(List<Integer> newData) {
		double[] inputs = new double[newData.size()];
		for (int i = 0; i < newData.size(); i++) {
			inputs[i] = newData.get(i);
		}
		double[] outputs = getOutputs(inputs);
		Map<L, Double> labelOutputMap = new HashMap<L, Double>();
		for (int i = 0; i < outputs.length; i++) {
			labelOutputMap.put(indexToLabel.get(i), outputs[i]);
		}
		return Utils.getHighestScorer(labelOutputMap);
	}
	
	private double[] getOutputs(double[] newData) {
		double[] currentLayerInputs = newData;
		for (int i = 0; i < layers.size(); i++) {
			List<NonLinearUnit> nextLayer = layers.get(i);
			double[] newLayerOutputs = new double[nextLayer.size()];
			for (int j = 0; j < newLayerOutputs.length; j++) {
				newLayerOutputs[j] = nextLayer.get(i).output(currentLayerInputs);
			}
			currentLayerInputs = newLayerOutputs;
		}
		return currentLayerInputs;
	}
	
	public void train() {
		
	}
}

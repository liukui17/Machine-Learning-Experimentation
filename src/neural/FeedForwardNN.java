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
			inputLayer.add(new NonLinearUnit(dimensionality));
		}
		for (int i = 1; i < layerSizes.length; i++) {
			List<NonLinearUnit> nextLayer = new ArrayList<NonLinearUnit>(layerSizes[i]);
			for (int j = 0; j < layerSizes[i]; j++) {
				nextLayer.add(new NonLinearUnit(layers.get(i - 1).size()));
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
	
	/**
	 * compute sigmoid of some input
	 */
	public double sigmoid(double input) {
		return 1.0 / (1.0 + Math.exp(-input));
	}
	
	/**
	 * compute derivative of sigmoid at input
	 */
	public double dSigmoid(double input) {
		double sig = sigmoid(input);
		return sig * (1.0 - sig);
	}
	
	private class NonLinearUnit {
		
		double[] weights;
		
		public NonLinearUnit(int inputSize) {
			weights = new double[inputSize + 1];
			initializeRandomWeights();
		}
		
		public void initializeRandomWeights() {
			Random random = new Random();
			for (int i = 0; i < weights.length; i++) {
				if (random.nextBoolean()) {
					weights[i] = random.nextDouble() / 20;
				} else {
					weights[i] = -random.nextDouble() / 20;
				}
			}
		}
		
		public double output(double[] data) {
			return sigmoid(computeLinearOutput(data));
		}
		
		public double computeLinearOutput(double[] data) {
			double res = weights[0];
			for (int i = 0; i < data.length; i++) {
				res += weights[i + 1] * data[i];
			}
			return res;
		}
	}
}

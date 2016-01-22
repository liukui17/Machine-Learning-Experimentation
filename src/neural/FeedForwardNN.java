package neural;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import data.Instance;

public class FeedForwardNN<L> {
	
	double learningRate = 0.002;
	
	List<List<NonLinearUnit>> layers;
	
	List<Instance<Integer, L>> trainingExamples;
	
	int dimensionality;
	
	public FeedForwardNN(List<Instance<Integer, L>> trainingExamples, int[] layerSizes) {
		this.trainingExamples = trainingExamples;
		dimensionality = trainingExamples.get(0).getDimensionality();
		
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
		}
		
		public void initializeRandomWeights() {
			Random random = new Random();
			for (int i = 0; i < weights.length; i++) {
				if (random.nextBoolean()) {
					weights[i] = random.nextDouble();
				} else {
					weights[i] = -random.nextDouble();
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

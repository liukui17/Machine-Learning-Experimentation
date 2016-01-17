package neural;

import java.util.List;
import java.util.Random;

import data.Instance;

public class BinaryPerceptron {
	
	static final int LEARNING_RATE_DECAY = 4;
	
//	double errorOnTrainingData;
	
	int steps;
	
	double learningRate;
	
	double[] weights;
	
	List<Instance<Double, Boolean>> trainingExamples;

	public BinaryPerceptron(List<Instance<Double, Boolean>> trainingExamples, int steps) {
		setUp(trainingExamples, steps);
		train();
	}
	
	private void setUp(List<Instance<Double, Boolean>> trainingExamples, int steps) {
		this.trainingExamples = trainingExamples;
		this.steps = steps;
		weights = new double[trainingExamples.get(0).getDimensionality() + 1];
	//	errorOnTrainingData = 0;
		initializeRandomWeights();
		learningRate = 0.1;
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
	
	public Boolean predict(List<Double> data) {
		double res = computeLinearOutput(data);
		if (res > 0.0) {
			return true;
		} else {
			return false;
		}
	}
	
	private double computeLinearOutput(List<Double> data) {
		double res = weights[0];
		for (int i = 0; i < data.size(); i++) {
			res += weights[i + 1] * data.get(i);
		}
		return res;
	}
	
	public void train() {
		for (int i = 0; i < steps; i++) {
			subsetGradientDescentUpdate(trainingExamples);
			learningRate -= learningRate / LEARNING_RATE_DECAY;
		}
	}
	
	private void subsetGradientDescentUpdate(List<Instance<Double, Boolean>> subset) {
		double[] updates = new double[weights.length];
		for (Instance<Double, Boolean> instance : subset) {
			double expected = 1.0;
			if (!instance.getLabel()) {
				expected = -1.0;
			}
			boolean prediction = predict(instance.getAttributeValues());
			double res = 1.0;
			if (!prediction) {
				res = -1.0;
			}
			updates[0] += learningRate * (expected - res);
			for (int i = 0; i < instance.getDimensionality(); i++) {
				updates[i + 1] += learningRate * (expected - res) * instance.getAttributeValue(i);
			}
		}
		for (int i = 0; i < weights.length; i++) {
			weights[i] += updates[i];
		}
	}
	
	public void printWeights() {
		System.out.print("[" + weights[0]);
		for (int i = 1; i < weights.length; i++) {
			System.out.print(", " + weights[i]);
		}
		System.out.println();
	}
}

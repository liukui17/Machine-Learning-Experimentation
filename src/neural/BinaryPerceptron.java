package neural;

import java.util.List;
import java.util.Random;

import data.Instance;

public class BinaryPerceptron {
	
	static final double LEARNING_RATE_DECAY = 0.99;
	
	static final double STOCHASTIC_LEARNING_RATE_DECAY = 0.9999;
	
	double errorOnTrainingData;
	
	int steps;
	
	double learningRate = 0.002;
	
	double[] weights;
	
	List<Instance<Integer, Boolean>> trainingExamples;
	
	public BinaryPerceptron(List<Instance<Integer, Boolean>> trainingExamples, int threshold) {
		setUp(trainingExamples);
		steps = 0;
	//	train(threshold);
		stochasticTrain(threshold);
	}
	
	private void setUp(List<Instance<Integer, Boolean>> trainingExamples) {
		this.trainingExamples = trainingExamples;
		weights = new double[trainingExamples.get(0).getDimensionality() + 1];
		errorOnTrainingData = 0;
	//	initializeRandomWeights();
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
	
	public boolean predict(List<Integer> data) {
		double res = computeUnthresholdedOutput(data);
		if (res > 0.0) {
			return true;
		} else {
			return false;
		}
	}
	
	private double computeUnthresholdedOutput(List<Integer> data) {
		double res = weights[0];
		for (int i = 0; i < data.size(); i++) {
			res += weights[i + 1] * data.get(i);
		}
		return res;
	}
	
	public void train(int threshold) {
		while (getErrorCountOnTrainingData() > threshold && learningRate > 0.00000001) {
		//	perceptronUpdate(trainingExamples);
			subsetGradientDescentUpdate(trainingExamples);
			learningRate *= LEARNING_RATE_DECAY;
			steps++;
		}
		errorOnTrainingData = ((double) getErrorCountOnTrainingData()) / trainingExamples.size();
	}
	
	public void stochasticTrain(int threshold) {
		while (learningRate > 0.00000001) {
			stochasticGradientDescentUpdate(trainingExamples.get(steps % trainingExamples.size()));
			learningRate *= STOCHASTIC_LEARNING_RATE_DECAY;
			steps++;
		}
	}
	
	private int getErrorCountOnTrainingData() {
		int count = 0;
		for (Instance<Integer, Boolean> instance : trainingExamples) {
			if (predict(instance.getAttributeValues()) != instance.getLabel()) {
				count++;
			}
		}
		return count;
	}
	
	private void stochasticGradientDescentUpdate(Instance<Integer, Boolean> example) {
		double expected = 1.0;
		if (!example.getLabel()) {
			expected = -1.0;
		}
		double res = computeUnthresholdedOutput(example.getAttributeValues());
		weights[0] += learningRate * (expected - res);
		for (int i = 0; i < example.getDimensionality(); i++) {
			weights[i + 1] += (learningRate * (expected - res) * example.getAttributeValue(i));
		}
	}
	
	private void subsetGradientDescentUpdate(List<Instance<Integer, Boolean>> subset) {
		double[] updates = new double[weights.length];
		for (Instance<Integer, Boolean> instance : subset) {
			double expected = 1.0;
			if (!instance.getLabel()) {
				expected = -1.0;
			}
			double res = computeUnthresholdedOutput(instance.getAttributeValues());
		//	System.out.println(res);
			updates[0] += learningRate * (expected - res);
			for (int i = 0; i < instance.getDimensionality(); i++) {
				updates[i + 1] += (learningRate * (expected - res) * instance.getAttributeValue(i));
			}
		}
		for (int i = 0; i < weights.length; i++) {
			weights[i] += updates[i];
		}
	}
	
	private void perceptronUpdate(List<Instance<Integer, Boolean>> subset) {
		double[] updates = new double[weights.length];
		for (Instance<Integer, Boolean> instance : subset) {
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
				updates[i + 1] += (learningRate * (expected - res) * instance.getAttributeValue(i));
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
		System.out.println("]");
	}
	
	public double getTrainingError() {
		return errorOnTrainingData;
	}
	
	public int getTrainingStepCount() {
		return steps;
	}
}

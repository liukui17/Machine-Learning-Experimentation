package neural;

import java.util.List;
import java.util.Random;

import data.Instance;

public class BinaryPerceptron {
	
	static final double LEARNING_RATE_DECAY = 0.99;
	
	double errorOnTrainingData;
	
	int steps;
	
	double learningRate = 0.1;
	
	double[] weights;
	
	List<Instance<Integer, Boolean>> trainingExamples;
	
	public BinaryPerceptron(List<Instance<Integer, Boolean>> trainingExamples, int threshold) {
		setUp(trainingExamples);
		steps = 0;
		train(threshold);
	}
	
	private void setUp(List<Instance<Integer, Boolean>> trainingExamples) {
		this.trainingExamples = trainingExamples;
		weights = new double[trainingExamples.get(0).getDimensionality() + 1];
		errorOnTrainingData = 0;
		initializeRandomWeights();
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
	
	public Boolean predict(List<Integer> data) {
		double res = computeLinearOutput(data);
		if (res > 0.0) {
			return true;
		} else {
			return false;
		}
	}
	
	private double computeLinearOutput(List<Integer> data) {
		double res = weights[0];
		for (int i = 0; i < data.size(); i++) {
			res += weights[i + 1] * data.get(i);
		}
		return res;
	}
	
/*	public void train() {
		for (int i = 0; i < steps; i++) {
			subsetGradientDescentUpdate(trainingExamples);
			learningRate *= LEARNING_RATE_DECAY;
		}
	} */
	
	public void train(int threshold) {
		while (getErrorCountOnTrainingData() > threshold && learningRate > 0.000001) {
			subsetGradientDescentUpdate(trainingExamples);
			learningRate *= LEARNING_RATE_DECAY;
			steps++;
		}
		errorOnTrainingData = ((double) getErrorCountOnTrainingData()) / trainingExamples.size();
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
	
	private void subsetGradientDescentUpdate(List<Instance<Integer, Boolean>> subset) {
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

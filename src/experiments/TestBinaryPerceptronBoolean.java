package experiments;

import java.util.List;
import java.util.Random;

import data.Instance;
import data.Utils;
import neural.BinaryPerceptron;

public class TestBinaryPerceptronBoolean {
	static final Random RANDOM = new Random();
	static final int TRIALS = 10;
	static final int DIMENSIONS = 16;
	static final int TRAIN_SIZE = 128;
	static final int TEST_SIZE = 1024;
	static final int BOOLEAN_FUNCTION = 0;

	public static void main(String[] args) {
		for (int j = 0; j < TRIALS; j++) {
			System.out.println("===== TEST " + j + " =====");
			List<Instance<Double, Boolean>> trainInstances = Utils.DoubleAttributeBooleanInstances(TRAIN_SIZE, DIMENSIONS, BOOLEAN_FUNCTION);
			List<Instance<Double, Boolean>> testInstances = Utils.DoubleAttributeBooleanInstances(TEST_SIZE, DIMENSIONS, BOOLEAN_FUNCTION);
			
			long start = System.currentTimeMillis();
			BinaryPerceptron classifier = new BinaryPerceptron(trainInstances, 2);
			System.out.println("Training complete (" + (System.currentTimeMillis() - start) + " milliseconds)");
	
			int correct = 0;
			for (int i = 0; i < testInstances.size(); i++) {
				boolean predicted = classifier.predict(testInstances.get(i).getAttributeValues());
				boolean actual = testInstances.get(i).getLabel();
				if (predicted == actual) {
					correct++;
				}
			}
			System.out.println("Correct: " + correct + "\tTotal: " + testInstances.size());
			classifier.printWeights();
		}
	}
}

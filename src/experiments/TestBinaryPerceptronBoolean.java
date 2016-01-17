package experiments;

import java.util.List;
import java.util.Random;

import data.Instance;
import data.Utils;
import neural.BinaryPerceptron;

/**
 * The perceptron really quickly learns to classify instances of
 * AND and OR. It fails on XOR (makes sense since XOR isn't linearly
 * separable). It can seem to error on AND and OR on tests but these
 * trials are misleading since what very likely happened was that the
 * training data didn't contain any instances of 'true' (for AND) and
 * 'false' (for OR)
 */
public class TestBinaryPerceptronBoolean {
	static final Random RANDOM = new Random();
	static final int TRIALS = 10;
	static final int DIMENSIONS = 3;
	static final int TRAIN_SIZE = 32;
	static final int TEST_SIZE = 32;
	static final int BOOLEAN_FUNCTION = 0;

	public static void main(String[] args) {
		for (int j = 0; j < TRIALS; j++) {
			System.out.println("===== TEST " + j + " =====");
			List<Instance<Double, Boolean>> trainInstances = Utils.DoubleAttributeBooleanInstances(TRAIN_SIZE, DIMENSIONS, BOOLEAN_FUNCTION);
			List<Instance<Double, Boolean>> testInstances = Utils.DoubleAttributeBooleanInstances(TEST_SIZE, DIMENSIONS, BOOLEAN_FUNCTION);
			
		/*	for (int i = 0; i < trainInstances.size(); i++) {
				System.out.println(trainInstances.get(i));
			}
			for (int i = 0; i < testInstances.size(); i++) {
				System.out.println(testInstances.get(i));
			} */
			
			long start = System.currentTimeMillis();
			BinaryPerceptron classifier = new BinaryPerceptron(trainInstances, 0);
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

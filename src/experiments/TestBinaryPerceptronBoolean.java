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
 * 'false' (for OR).
 * 
 * The perceptron learns fairly quickly to classify at-least-M-of-N rules.
 * It a lower learning however or else it will diverge and the weights
 * blow up. Another mistake was the perceptron wasn't given enough
 * data so it was learning incorrect concepts.
 * 
 * Using 4096 at-least-M-of-N examples (M = N/2, N = 20), we can get pretty
 * much 100% classification accuracy (1024 testing examples) on all trials.
 * It takes around 150 milliseconds to train and around 350 steps to converge.
 */
public class TestBinaryPerceptronBoolean {
	static final Random RANDOM = new Random();
	static final int TRIALS = 10;
	static final int DIMENSIONS = 20;
	static final int TRAIN_SIZE = 4096;
	static final int TEST_SIZE = 1024;
	static final int BOOLEAN_FUNCTION = 0;

	public static void main(String[] args) {
		for (int j = 0; j < TRIALS; j++) {
			System.out.println("===== TEST " + j + " =====");
		//	List<Instance<Integer, Boolean>> trainInstances = Utils.IntegerAttributeBooleanInstances(TRAIN_SIZE, DIMENSIONS, BOOLEAN_FUNCTION);
		//	List<Instance<Integer, Boolean>> testInstances = Utils.IntegerAttributeBooleanInstances(TEST_SIZE, DIMENSIONS, BOOLEAN_FUNCTION);
			List<Instance<Integer, Boolean>> trainInstances = Utils.IntegerAttributeMOfNInstances(TRAIN_SIZE, DIMENSIONS / 2, DIMENSIONS, true);
			List<Instance<Integer, Boolean>> testInstances = Utils.IntegerAttributeMOfNInstances(TEST_SIZE, DIMENSIONS / 2, DIMENSIONS, true);
			
		/*	for (int i = 0; i < trainInstances.size(); i++) {
				System.out.println(trainInstances.get(i));
			}
			for (int i = 0; i < testInstances.size(); i++) {
				System.out.println(testInstances.get(i));
			} */
			
			long start = System.currentTimeMillis();
			BinaryPerceptron classifier = new BinaryPerceptron(trainInstances, 0);
			System.out.println("Training complete (" + (System.currentTimeMillis() - start) + " milliseconds)");
			System.out.println("Trained in " + classifier.getTrainingStepCount() + " steps and achieved " +
						(1 - classifier.getTrainingError()) + " accuracy on training data.");
	
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

package experiments;

import java.util.List;

import data.Instance;
import data.MNISTParser;
import data.Utils;
import neural.BinaryPerceptron;

public class TestBinaryPerceptronOneVSAll {
	static final int TRIALS = 10;
	static final int ODD_ONE_OUT = 1;
	
	public static void main(String[] args) {
		List<int[][]> trainingImages = MNISTParser.readImages(MNISTParser.TRAIN_IMAGES);
		int[] trainingLabels = MNISTParser.readLabels(MNISTParser.TRAIN_LABELS);
		List<int[][]> testImages = MNISTParser.readImages(MNISTParser.TEST_IMAGES);
		int[] testLabels = MNISTParser.readLabels(MNISTParser.TEST_LABELS);
		
		List<Instance<Integer, Boolean>> trainInstances = MNISTParser.makeOneVSAllInstances(trainingImages,
				trainingLabels, 0, 60000, ODD_ONE_OUT);
		List<Instance<Integer, Boolean>> testInstances = MNISTParser.makeOneVSAllInstances(testImages,
				testLabels, 0, 10000, ODD_ONE_OUT);
		
		trainingImages = null;
		trainingLabels = null;
		testImages = null;
		testLabels = null;
		
		List<Instance<Integer, Boolean>> subTrainInstances = Utils.makeSubset(trainInstances, 10000);
		
		for (int j = 0; j < TRIALS; j++) {
			System.out.println("===== TEST " + j + " =====");
			long start = System.currentTimeMillis();
			BinaryPerceptron classifier = new BinaryPerceptron(subTrainInstances, 0);
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
		}
	}
}

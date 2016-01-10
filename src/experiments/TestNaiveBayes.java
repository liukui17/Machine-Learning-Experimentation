package experiments;

import java.util.List;

import bayes.NaiveBayesClassifier;
import data.Instance;
import data.MNISTParser;

/**
 * Trained on all 60000 examples, the naive Bayes learner gets 84.14%
 * classification accuracy and takes around 30 seconds to train.
 */
public class TestNaiveBayes {
	public static void main(String[] args) {
		List<int[][]> trainingImages = MNISTParser.readImages(MNISTParser.TRAIN_IMAGES);
		int[] trainingLabels = MNISTParser.readLabels(MNISTParser.TRAIN_LABELS);
		List<int[][]> testImages = MNISTParser.readImages(MNISTParser.TEST_IMAGES);
		int[] testLabels = MNISTParser.readLabels(MNISTParser.TEST_LABELS);
		
		List<Instance<Integer,Integer>> trainInstances = MNISTParser.makeInstances(trainingImages, trainingLabels, 0, trainingImages.size());
		List<Instance<Integer,Integer>> testInstances = MNISTParser.makeInstances(testImages, testLabels, 0, testImages.size());
		
		trainingImages = null;
		trainingLabels = null;
		testImages = null;
		testLabels = null;

		long start = System.currentTimeMillis();
		NaiveBayesClassifier<Integer, Integer> classifier = new NaiveBayesClassifier<Integer, Integer>(trainInstances);
		System.out.println("Training complete (" + (System.currentTimeMillis() - start) + " milliseconds)");
		
		int correct = 0;
		int total = testInstances.size();
		for (int i = 0; i < total; i++) {
			Instance<Integer,Integer> nextInstance = testInstances.get(i);
			int predicted = classifier.predict(nextInstance.getAttributeValues());
			int actual = nextInstance.getLabel();
			if (predicted == actual) {
				correct++;
			}
		}
		System.out.println("Correct: " + correct + "\tTotal: " + total);
	}
}

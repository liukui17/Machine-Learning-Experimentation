package experiments;

import java.util.List;

import data.Instance;
import data.MNISTParser;
import data.Utils;
import neighbors.KNNDecisionTree;
import neighbors.KNNRandomForest;

public class TestKNNTrees {
	public static void main(String[] args) {
		List<int[][]> trainingImages = MNISTParser.readImages(MNISTParser.TRAIN_IMAGES);
		int[] trainingLabels = MNISTParser.readLabels(MNISTParser.TRAIN_LABELS);
		List<int[][]> testImages = MNISTParser.readImages(MNISTParser.TEST_IMAGES);
		int[] testLabels = MNISTParser.readLabels(MNISTParser.TEST_LABELS);

		List<Instance<Integer, Integer>> trainInstances = MNISTParser.makeInstances(trainingImages, trainingLabels, 0,
				trainingImages.size());
		List<Instance<Integer, Integer>> testInstances = MNISTParser.makeInstances(testImages, testLabels, 0,
				testImages.size());
		
		List<Instance<Integer, Integer>> subTrainInstances = Utils.makeSubset(trainInstances, 60000);

		/*
		 * Allow the garbage collector to free up some space on the heap since
		 * the training and testing instances have already been made.
		 */
		trainingImages = null;
		trainingLabels = null;
		testImages = null;
		testLabels = null;
		
		long start = System.currentTimeMillis();
		KNNDecisionTree<Integer> classifier = new KNNDecisionTree<Integer>(subTrainInstances, 100, 0.000001, 2, 10);
	//	KNNRandomForest<Integer> classifier = new KNNRandomForest<Integer>(trainInstances, 61, 0.00001, 2, 10);
		System.out.println("Training complete (" + (System.currentTimeMillis() - start) + " milliseconds)");
		
		int correct = 0;
		int total = testInstances.size();
		start = System.currentTimeMillis();
		for (int i = 0; i < total; i++) {
			Instance<Integer, Integer> nextInstance = testInstances.get(i);
			int predicted = classifier.predict(nextInstance.getAttributeValues());
			int actual = nextInstance.getLabel();
			if (predicted == actual) {
				correct++;
			}
		}
		System.out.println("Correct: " + correct + "\tTotal: " + total +
				"\tPrediction Time: " + (System.currentTimeMillis() - start));
	}
}

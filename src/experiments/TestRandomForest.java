package experiments;
import java.util.ArrayList;
import java.util.List;

import data.Instance;
import data.MNISTParser;
import data.Utils;
import decision_tree.*;

/**
 * A test class for the RandomForest implementation.
 * 
 * Training a RandomForest using a committee of 100 DecisionTrees, each with
 * a randomized subset of the original 60000 examples of size 40000, takes around
 * 75 minutes and gets around 95.41% prediction accuracy.
 */
public class TestRandomForest {
	
	public static void main(String[] args) {
	//	System.out.println("Reading data...");
		List<int[][]> trainingImages = MNISTParser.readImages(MNISTParser.TRAIN_IMAGES);
		int[] trainingLabels = MNISTParser.readLabels(MNISTParser.TRAIN_LABELS);
		List<int[][]> testImages = MNISTParser.readImages(MNISTParser.TEST_IMAGES);
		int[] testLabels = MNISTParser.readLabels(MNISTParser.TEST_LABELS);
		
		List<Instance<Integer,Integer>> trainingExamples = MNISTParser.makeInstances(trainingImages, trainingLabels, 0, trainingImages.size());
		List<Instance<Integer,Integer>> testInstances = MNISTParser.makeInstances(testImages, testLabels, 0, testImages.size());
		
		int trainSubsetSize = (trainingExamples.size() / 6) * 4;
		
		for (int j = 0; j < 1; j++) {
			System.out.println("===== TEST " + j + " =====");
			List<List<Instance<Integer,Integer>>> trainingSets = new ArrayList<List<Instance<Integer,Integer>>>(100);
			for (int i = 0; i < 100; i++) {
				trainingSets.add(Utils.makeSubset(trainingExamples, trainSubsetSize));
			}
			
			System.out.println("Training...");
			long startTime = System.currentTimeMillis();
			RandomForest<Integer,Integer> forest = new RandomForest<Integer,Integer>(trainingSets);
			System.out.println("Training complete (" + (System.currentTimeMillis() - startTime) + " milliseconds)");
			
			int correct = 0;
			for (int i = 0; i < testInstances.size(); i++) {
				Instance<Integer,Integer> nextInstance = testInstances.get(i);
				int predicted = forest.predict(nextInstance.getAttributeValues());
				int actual = nextInstance.getLabel();
				if (predicted == actual) {
					correct++;
				}
			}
			System.out.println("Correct: " + correct + "\tTotal: " + testInstances.size());
		}
	}
}

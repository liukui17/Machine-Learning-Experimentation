package experiments;

import java.util.List;

import data.Instance;
import data.MNISTParser;
import data.SD19Reader;
import data.Utils;
import decision_tree.*;

/**
 * A test class for the DecisionTree implementation.
 * 
 * On all 60000 training examples, the DecisionTree gets around 89.23%
 * prediction accuracy and takes around 1.5 minutes to train.
 */
public class TestDecisionTree {
	public static void main(String[] args) {
	//	System.out.println("Reading data...");
		List<int[][]> trainingImages = MNISTParser.readImages(MNISTParser.TRAIN_IMAGES);
		int[] trainingLabels = MNISTParser.readLabels(MNISTParser.TRAIN_LABELS);
		List<int[][]> testImages = MNISTParser.readImages(MNISTParser.TEST_IMAGES);
		int[] testLabels = MNISTParser.readLabels(MNISTParser.TEST_LABELS);
		
	/*	List<BooleanInstance<Character>> allInstances = SD19Reader.readSD19Instances(SD19Reader.SD19_FILE);
		System.out.println(allInstances.get(0).getLabel());
		for (int i = 0; i < SD19Reader.IMAGE_HEIGHT; i++) {
			for (int j = 0; j < SD19Reader.IMAGE_WIDTH; j++) {
				boolean next = allInstances.get(0).getAttributeValue(i * SD19Reader.IMAGE_WIDTH + j);
				if (next) {
					System.out.print(1);
				} else {
					System.out.print(0);
				}
			}
			System.out.println();
		} */
		
	//	printImage(images.get(2));
	//	System.out.println(Arrays.toString(labels));
		List<Instance<Integer,Integer>> trainInstances = MNISTParser.makeInstances(trainingImages, trainingLabels, 0, trainingImages.size());
		List<Instance<Integer,Integer>> testInstances = MNISTParser.makeInstances(testImages, testLabels, 0, testImages.size());
		
		int trainSubsetSize = trainInstances.size();
		
		for (int j = 0; j < 1; j++) {
			System.out.println("===== TEST " + j + " =====");
			List<Instance<Integer,Integer>> subTrainInstances = Utils.makeSubset(trainInstances, trainSubsetSize);
			
			System.out.println("Training...");
			long startTime = System.currentTimeMillis();
			DecisionTree<Integer,Integer> tree = new DecisionTree<Integer,Integer>(subTrainInstances);
			System.out.println("Training complete (" + (System.currentTimeMillis() - startTime) + " milliseconds)");
			
			int correct = 0;
			for (int i = 0; i < testInstances.size(); i++) {
				Instance<Integer,Integer> nextInstance = testInstances.get(i);
				int predicted = tree.predict(nextInstance.getAttributeValues());
				int actual = nextInstance.getLabel();
				if (predicted == actual) {
					correct++;
				}
			}
			System.out.println("Correct: " + correct + "\tTotal: " + testInstances.size());
		}
		
		/*	Scanner in = new Scanner(System.in);
		String next = null;
		while ((next = in.nextLine()) != null && !next.equals("quit")) {
			int nextImageSelection = Integer.parseInt(next);
			Instance<Integer,Integer> nextInstance = testInstances.get(nextImageSelection % testInstances.size());
			int nextPrediction = tree.predict(nextInstance.getAttributeValues());
			int nextTrueLabel = nextInstance.getLabel();
			System.out.println("Predicted: " + nextPrediction + "\tActual: " + nextTrueLabel);
		} */
	}
	
	public static void printImage(int[][] image) {
		for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[i].length; j++) {
				if (image[i][j] != 0) {
					System.out.print("1");
				} else {
					System.out.print("0");
				}
			}
			System.out.println();
		}
	}
}

package experiments;

import java.util.List;

import clustering.*;
import data.MNISTParser;
import data.NumericalInstance;

public class TestKNN {

	public static void main(String[] args) {
		List<int[][]> trainingImages = MNISTParser.readImages(MNISTParser.TRAIN_IMAGES);
		int[] trainingLabels = MNISTParser.readLabels(MNISTParser.TRAIN_LABELS);
		List<int[][]> testImages = MNISTParser.readImages(MNISTParser.TEST_IMAGES);
		int[] testLabels = MNISTParser.readLabels(MNISTParser.TEST_LABELS);
		
		List<NumericalInstance<Integer>> trainInstances = MNISTParser.makeNumericalInstances(trainingImages, trainingLabels, 0, trainingImages.size());
		List<NumericalInstance<Integer>> testInstances = MNISTParser.makeNumericalInstances(testImages, testLabels, 0, testImages.size());
		
		KNearestNeighborsRegression<Integer> knn = new KNearestNeighborsRegression<Integer>(trainInstances, 10, 0);
		
		int correct = 0;
		long start = System.currentTimeMillis();
		for (int i = 0; i < 100; i++) {
			int prediction = knn.predict(testInstances.get(i).getAttributeValues());
			int actual = testInstances.get(i).getLabel();
			if (prediction == actual){
				correct++;
			}
		}
		System.out.println("Correct: " + correct + "\tTotal: " + testInstances.size() +
						   "\tTime: " + (System.currentTimeMillis() - start) + " milliseconds");
	}
}

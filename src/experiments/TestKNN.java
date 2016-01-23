package experiments;
 
import java.util.List;

import data.Instance;
import data.MNISTParser;
import neighbors.KNearestNeighborsClassifier;
 
 public class TestKNN {
 
 	public static void main(String[] args) {
 		List<int[][]> trainingImages = MNISTParser.readImages(MNISTParser.TRAIN_IMAGES);
 		int[] trainingLabels = MNISTParser.readLabels(MNISTParser.TRAIN_LABELS);
 		List<int[][]> testImages = MNISTParser.readImages(MNISTParser.TEST_IMAGES);
 		int[] testLabels = MNISTParser.readLabels(MNISTParser.TEST_LABELS);
 		
 		List<Instance<Integer, Integer>> trainInstances = MNISTParser.makeInstances(trainingImages, trainingLabels, 0,
				trainingImages.size());
		List<Instance<Integer, Integer>> testInstances = MNISTParser.makeInstances(testImages, testLabels, 0,
				testImages.size());
 		
 		KNearestNeighborsClassifier<Integer> knn = new KNearestNeighborsClassifier<Integer>(trainInstances, 10, 2);
 		
 		int correct = 0;
 		int total = 10000;
 		long start = System.currentTimeMillis();
 		for (int i = 0; i < total; i++) {
 			int prediction = knn.predict(testInstances.get(i).getAttributeValues());
 			int actual = testInstances.get(i).getLabel();
 			if (prediction == actual){
 				correct++;
 			}
 		}
 		System.out.println("Correct: " + correct + "\tTotal: " + total +
 						   "\tTime: " + (System.currentTimeMillis() - start) + " milliseconds");
 	}
 }
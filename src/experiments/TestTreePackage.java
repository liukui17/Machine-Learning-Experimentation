package experiments;

import java.util.List;

import data.Instance;
import data.MNISTParser;
import data.Utils;
import tree.*;

/**
 * General class for testing and experimenting with the classifiers in the tree
 * package.
 * 
 * EntropyDecisionTree: with an EntropyDecisionTree trained on 60000 examples,
 * the classifier gets around 89.23% classification accuracy and takes around 75
 * seconds to train
 * 
 * EntropyRandomForest: with a committee of 100 EntropyDecisionTrees, each
 * trained on a randomized subset of the 60000 training examples of size 40000,
 * the classifier gets around 95.25% (highest 95.41%) classification accuracy
 * and takes around 75 minutes to train
 * 
 * RandomDecisionTree: with a RandomDecisionTree trained on 60000 examples, the
 * classifier gets around 74 (highest 75.44%) classification accuracy and takes
 * around 500-700 milliseconds to train
 * 
 * ExtremelyRandomForest: with a committee of 60 RandomDecisionTrees, each
 * trained with 60000 training examples, the classifier gets around 95.75%
 * (highest 95.93%) classification accuracy and takes around 1.5 minutes to
 * train (can't train using larger committees due to memory constraints)
 * 
 * Discussion:
 * 
 * The advantages of RandomDecisionTrees over EntropyDecisionTrees are that they
 * train really quickly. The problem is they easily overfit since there is no
 * particular reason they choose to split on any given attribute. Further, since
 * the attributes are randomly chosen, it takes longer to make propagated
 * subsets pure, leading to larger trees. In particular, RandomDecisionTrees
 * have much higher memory usage and also tend to overfit by learning
 * extraneous/false rules.
 * 
 * ExtremelyRandomForests, like EntropyDecisionTrees, train much more quickly
 * than EntropyRandomForests. However, they use a huge amount of memory since
 * each tree is a RandomDecisionTree. For some reason, they also tend to have
 * slightly higher classification accuracy than EntropyRandomForests. They do
 * address the overfitting problem of RandomDecisionTrees but I'm not sure why
 * they would actually do better. Perhaps it's because each tree
 * RandomDecisionTree was trained on all 60000 training examples while each tree
 * of the EntropyRandomForest was trained on only 40000 (using all 60000
 * examples would lead to a committee of all identical trees).
 */
public class TestTreePackage {
	public static void main(String[] args) {
		List<int[][]> trainingImages = MNISTParser.readImages(MNISTParser.TRAIN_IMAGES);
		int[] trainingLabels = MNISTParser.readLabels(MNISTParser.TRAIN_LABELS);
		List<int[][]> testImages = MNISTParser.readImages(MNISTParser.TEST_IMAGES);
		int[] testLabels = MNISTParser.readLabels(MNISTParser.TEST_LABELS);

		List<Instance<Integer, Integer>> trainInstances = MNISTParser.makeInstances(trainingImages, trainingLabels, 0,
				trainingImages.size());
		List<Instance<Integer, Integer>> testInstances = MNISTParser.makeInstances(testImages, testLabels, 0,
				testImages.size());

		/*
		 * Allow the garbage collector to free up some space on the heap since
		 * the training and testing instances have already been made.
		 */
		trainingImages = null;
		trainingLabels = null;
		testImages = null;
		testLabels = null;

		int trainSubsetSize = trainInstances.size();

		for (int j = 0; j < 1; j++) {
			System.out.println("===== TEST " + j + " =====");
			List<Instance<Integer, Integer>> subTrainInstances = Utils.makeSubset(trainInstances, trainSubsetSize);

			System.out.println("Training...");
			long startTime = System.currentTimeMillis();
		//	DecisionTree<Integer,Integer> tree = new EntropyDecisionTree<Integer,Integer>(subTrainInstances);
		//	DecisionTree<Integer,Integer> tree = new RandomEntropyDecisionTree<Integer,Integer>(subTrainInstances, 100);
			DecisionTree<Integer,Integer> tree = new RandomDecisionTree<Integer,Integer>(subTrainInstances);
		//	RandomForest<Integer, Integer> forest = new ExtremelyRandomForest<Integer, Integer>(subTrainInstances, 61);
		//	RandomForest<Integer,Integer> forest = new EntropyRandomForest<Integer,Integer>(subTrainInstances, 40000, 101);
			System.out.println("Training complete (" + (System.currentTimeMillis() - startTime) + " milliseconds)");
			tree.printStats();
		//	forest.printStats();

			int correct = 0;
			for (int i = 0; i < testInstances.size(); i++) {
				Instance<Integer, Integer> nextInstance = testInstances.get(i);
				int predicted = tree.predict(nextInstance.getAttributeValues());
			//	int predicted = forest.predict(nextInstance.getAttributeValues());
				int actual = nextInstance.getLabel();
				if (predicted == actual) {
					correct++;
				}
			}
			System.out.println("Correct: " + correct + "\tTotal: " + testInstances.size());
		}
	}
}

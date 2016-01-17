package experiments;

import java.util.List;
import java.util.Random;

import data.Instance;
import data.Utils;
import decision_tree.DecisionTree;

public class TestDecisionTreeBooleanFunctions {
	static final Random RANDOM = new Random();
	static final int DIMENSIONS = 8;
	static final int TRAIN_SIZE = 32;
	static final int BOOLEAN_FUNCTION = 0;
	
	public static void main(String[] args) {
		List<Instance<Boolean,Boolean>> andTrain = Utils.BooleanInstances(TRAIN_SIZE, DIMENSIONS, BOOLEAN_FUNCTION);
		DecisionTree<Boolean,Boolean> tree = new DecisionTree<Boolean,Boolean>(andTrain);
		
		List<Instance<Boolean,Boolean>> andTest = Utils.BooleanInstances(TRAIN_SIZE, DIMENSIONS, BOOLEAN_FUNCTION);
		int correct = 0;
		for (int i = 0; i < andTest.size(); i++) {
			boolean predicted = tree.predict(andTest.get(i).getAttributeValues());
			boolean actual = andTest.get(i).getLabel();
			if (predicted == actual) {
				correct++;
			}
		}
		System.out.println("Correct: " + correct + "\tTotal: " + andTest.size());
	}
}

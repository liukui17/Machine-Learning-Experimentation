package experiments;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import data.Instance;
import decision_tree.DecisionTree;

public class TestDecisionTreeBooleanFunctions {
	static final Random RANDOM = new Random();
	static final int DIMENSIONS = 8;
	static final int TRAIN_SIZE = 32;
	
	public static void main(String[] args) {
		List<Instance<Boolean,Boolean>> andTrain = ANDInstances();
		DecisionTree<Boolean,Boolean> tree = new DecisionTree<Boolean,Boolean>(andTrain);
		
		List<Instance<Boolean,Boolean>> andTest = ANDInstances();
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
	
	public static List<Instance<Boolean,Boolean>> ANDInstances() {
		List<Instance<Boolean,Boolean>> instances = new ArrayList<Instance<Boolean,Boolean>>(TRAIN_SIZE);
		for (int i = 0; i < TRAIN_SIZE; i++) {
			List<Boolean> nextAttributes = new ArrayList<Boolean>();
			boolean res = true;
			for (int j = 0; j < DIMENSIONS; j++) {
				boolean next = RANDOM.nextBoolean();
				res &= next;
				nextAttributes.add(next);
			}
			instances.add(new Instance<Boolean,Boolean>(nextAttributes, res));
		}
		return instances;
	}
}

package data;

public class IntegerInstanceDistancePair<L> implements Comparable<IntegerInstanceDistancePair<L>> {
	
	private Instance<Integer, L> instance;
	private double dist;
	
	public IntegerInstanceDistancePair(Instance<Integer, L> instance, double dist) {
		this.instance = instance;
		this.dist = dist;
	}
	
	public int compareTo(IntegerInstanceDistancePair<L> other) {
		if (this.dist > other.dist) {
			return -1;
		} else if (this.dist == other.dist) {
			return 0;
		} else {
			return 1;
		}
	}
	
	public Instance<Integer, L> getInstance() {
		return instance;
	}
	
	public double getDistance() {
		return dist;
	}
}

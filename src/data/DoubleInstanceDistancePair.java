package data;

public class DoubleInstanceDistancePair<L> implements Comparable<DoubleInstanceDistancePair<L>> {
	
	private Instance<Double, L> instance;
	private double dist;
	
	public DoubleInstanceDistancePair(Instance<Double, L> instance, double dist) {
		this.instance = instance;
		this.dist = dist;
	}
	
	public int compareTo(DoubleInstanceDistancePair<L> other) {
		if (this.dist > other.dist) {
			return -1;
		} else if (this.dist == other.dist) {
			return 0;
		} else {
			return 1;
		}
	}
	
	public Instance<Double, L> getInstance() {
		return instance;
	}
	
	public double getDistance() {
		return dist;
	}
}

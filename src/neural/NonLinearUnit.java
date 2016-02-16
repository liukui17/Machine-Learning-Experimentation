package neural;

public abstract class NonLinearUnit {
	
	private double[] weights;
	
	public NonLinearUnit(int inputSize) {
		weights = new double[inputSize + 1];
	}
	
	public double output(double[] data) {
		return activation(computeLinearOutput(data));
	}
	
	public abstract double activation(double input);
	
	public abstract double dActivation(double input);
	
	public double computeLinearOutput(double[] data) {
		double res = weights[0];
		for (int i = 0; i < data.length; i++) {
			res += weights[i + 1] * data[i];
		}
		return res;
	}
	
	public double[] getWeights() {
		return weights;
	}
	
	public int getWeightLength() {
		return weights.length;
	}
	
	public double getWeight(int index) {
		assert(index >= 0 && index < weights.length);
		return weights[index];
	}
	
	public void updateWeight(int index, double weight) {
		assert(index >= 0 && index < weights.length);
		weights[index] = weight;
	}
}

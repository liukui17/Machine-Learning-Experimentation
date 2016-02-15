package neural;

public abstract class NonLinearUnit {
	
	double[] weights;
	
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
}

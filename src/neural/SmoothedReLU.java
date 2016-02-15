package neural;

import data.Utils;

public class SmoothedReLU extends NonLinearUnit {
	
	public SmoothedReLU(int inputSize) {
		super(inputSize);
	}
	
	public double activation(double input) {
		return Utils.softplus(input);
	}
	
	public double dActivation(double input) {
		return Utils.sigmoid(input);
	}
}

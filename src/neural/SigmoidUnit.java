package neural;

import data.Utils;

public class SigmoidUnit extends NonLinearUnit {
	
	public SigmoidUnit(int inputSize) {
		super(inputSize);
	}
	
	public double activation(double input) {
		return Utils.sigmoid(input);
	}
	
	public double dActivation(double input) {
		return Utils.dSigmoid(input);
	}
}

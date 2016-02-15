package neural;

import data.Utils;

public class TanhUnit extends NonLinearUnit {
	
	public TanhUnit(int inputSize) {
		super(inputSize);
	}
	
	public double activation(double input) {
		return Math.tanh(input);
	}
	
	public double dActivation(double input) {
		return Utils.dTanh(input);
	}
}

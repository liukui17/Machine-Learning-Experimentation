package neural;

import data.Utils;

public class RectifiedLinearUnit extends NonLinearUnit {
	
	double threshold;
	
	public RectifiedLinearUnit(int inputSize) {
		this(inputSize, 0.0);
	}
	
	public RectifiedLinearUnit(int inputSize, double threshold) {
		super(inputSize);
		this.threshold = threshold;
	}
	
	public double activation(double input) {
		return Utils.rectifiedLinear(input, threshold);
	}
	
	public double dActivation(double input) {
		return Utils.dRectifiedLinear(input, threshold);
	}
}

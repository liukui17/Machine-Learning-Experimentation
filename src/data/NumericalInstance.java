package data;

public class NumericalInstance<L> {
	double[] attributeValues;
	L label;
	int dimensionality;
	
	public NumericalInstance(double[] attributeValues, L label) {
		this.attributeValues = attributeValues;
		this.label = label;
		this.dimensionality = this.attributeValues.length;
	}
	
	public NumericalInstance(int size, L label) {
		this.attributeValues = new double[size];
		this.label = label;
		this.dimensionality = size;
	}
	
	public void setAttributeValue(int index, double attributeValue) {
		assert(index >=0 && index < attributeValues.length);
		attributeValues[index] = attributeValue;
	}
	
	public double getAttributeValue(int index) {
		assert(index >=0 && index < attributeValues.length);
		return attributeValues[index];
	}
	
	public double[] getAttributeValues() {
		return attributeValues;
	}
	
	public int getDimensionality() {
		return dimensionality;
	}
	
	public L getLabel() {
		return label;
	}
	
	public double LnNorm(NumericalInstance<L> other, int n) {
		assert(getDimensionality() == other.getDimensionality());
		double res = 0.0;
		for (int i = 0; i < attributeValues.length; i++) {
			res += Math.abs(Math.pow(attributeValues[i] - other.attributeValues[i], n));
		}
		return Math.pow(res, 1.0 / n);
	}
	
	public double LInfinityNorm(NumericalInstance<L> other) {
		assert(getDimensionality() == other.getDimensionality());
		double res = 0.0;
		for (int i = 0; i < attributeValues.length; i++) {
			double nextAttributeDistance = Math.abs(attributeValues[i] - other.attributeValues[i]);
			if (nextAttributeDistance > res) {
				res = nextAttributeDistance;
			}
		}
		return res;
	}
}

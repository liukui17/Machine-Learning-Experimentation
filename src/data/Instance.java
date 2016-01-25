package data;

import java.util.ArrayList;
import java.util.List;

public class Instance<A,L> {
	List<A> attributeValues;
	L label;
	int dimensionality;
	double weight;
	
	public Instance(List<A> attributeValues, L label, double weight) {
		this.attributeValues = attributeValues;
		this.label = label;
		this.weight = weight;
		this.dimensionality = this.attributeValues.size();
	}
	
	public Instance(List<A> attributeValues, L label) {
		this(attributeValues, label, 1.0);
	}
	
	public Instance(int size, L label) {
		this.attributeValues = new ArrayList<A>(size);
		this.label = label;
		this.dimensionality = size;
	}
	
	public void setAttributeValue(int index, A attributeValue) {
		assert(index >= 0 && index < attributeValues.size());
		attributeValues.set(index, attributeValue);
	}
	
	public A getAttributeValue(int index) {
		assert(index >=0 && index < attributeValues.size());
		return attributeValues.get(index);
	}
	
	public List<A> getAttributeValues() {
		return attributeValues;
	}
	
	public int getDimensionality() {
		return dimensionality;
	}
	
	public L getLabel() {
		return label;
	}
	
	public void setLabel(L newLabel) {
		label = newLabel;
	}
	
	public String toString() {
		return "{" + attributeValues.toString() + ", " + label.toString() + "}";
	}
}

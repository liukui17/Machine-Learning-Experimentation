package neural;

import java.util.ArrayList;
import java.util.List;

import data.NumericalInstance;

public class FeedForwardNN<L> {
	
	static final double LEARNING_RATE_DECAY = 0.99;
	
	static final double STOCHASTIC_LEARNING_RATE_DECAY = 0.9999;
	
	double learningRate = 0.002;
	
	List<L> indexToLabel;
	
	List<NonLinearUnit[]> layers;
	
	List<NumericalInstance<L>> trainingExamples;
	
	int dimensionality;
	
	double correctLabelTarget = 0.9;
	
	public FeedForwardNN(List<NumericalInstance<L>> trainingExamples, List<L> indexToLabel, int[] layerSizes) {
		this.trainingExamples = trainingExamples;
		dimensionality = trainingExamples.get(0).getDimensionality();
		this.indexToLabel = indexToLabel;
		
		layers = new ArrayList<NonLinearUnit[]>(layerSizes.length);
		NonLinearUnit[] inputLayer = new NonLinearUnit[layerSizes[0]];
		for (int i = 0; i < layerSizes[0]; i++) {
			inputLayer[i] = new SigmoidUnit(dimensionality);
		//	inputLayer.add(new SigmoidUnit(dimensionality));
		//	inputLayer.add(new TanhUnit(dimensionality));
		//	inputLayer.add(new RectifiedLinearUnit(dimensionality));
		}
		layers.add(inputLayer);
		for (int i = 1; i < layerSizes.length; i++) {
			NonLinearUnit[] nextLayer = new NonLinearUnit[layerSizes[i]];
			for (int j = 0; j < layerSizes[i]; j++) {
				nextLayer[i] = new SigmoidUnit(layers.get(i - 1).length);
			//	nextLayer.add(new SigmoidUnit(layers.get(i - 1).size()));
			//	nextLayer.add(new TanhUnit(layers.get(i - 1).size()));
			//	nextLayer.add(new RectifiedLinearUnit(layers.get(i - 1).size()));
			}
			layers.add(nextLayer);
		}
	}
	
	public L predict(double[] newData) {
		double[] outputs = getOutputs(newData);
		return indexToLabel.get(getLabelIndexFromOutput(outputs));
	}
	
	private int getLabelIndexFromOutput(double[] output) {
		int index = 0;
		double max = output[0];
		for (int i = 1; i < output.length; i++) {
			if (output[i] > max) {
				max = output[i];
				index = i;
			}
		}
		return index;
	}
	
	private double[] getOutputs(double[] newData) {
		double[] currentLayerInputs = newData;
		for (int i = 0; i < layers.size(); i++) {
			NonLinearUnit[] nextLayer = layers.get(i);
			double[] newLayerOutputs = new double[nextLayer.length];
			for (int j = 0; j < newLayerOutputs.length; j++) {
				newLayerOutputs[j] = nextLayer[i].output(currentLayerInputs);
			}
			currentLayerInputs = newLayerOutputs;
		}
		return currentLayerInputs;
	}
	
	private List<double[]> getAllLayerOutputs(double[] newData) {
		List<double[]> outputs = new ArrayList<double[]>(layers.size() + 1);
		double[] currentLayerInputs = newData;
		outputs.add(currentLayerInputs);
		for (int i = 0; i < layers.size(); i++) {
			NonLinearUnit[] nextLayer = layers.get(i);
			double[] newLayerOutputs = new double[nextLayer.length];
			for (int j = 0; j < newLayerOutputs.length; j++) {
				newLayerOutputs[j] = nextLayer[i].output(currentLayerInputs);
			}
			currentLayerInputs = newLayerOutputs;
			outputs.add(currentLayerInputs);
		}
		return outputs;
	}
	
	public void train() {
		int i = 0;
		while (learningRate > 0.00000001) {
			NumericalInstance<L> nextInstance = trainingExamples.get(i % trainingExamples.size());
			singleIterationBackProp(nextInstance);
			i++;
		}
	}
	
	private void singleIterationBackProp(NumericalInstance<L> instance) {
		List<double[]> outputs = getAllLayerOutputs(instance.getAttributeValues());
		int index = getLabelIndexFromOutput(outputs.get(outputs.size() - 1));
		List<double[]> allLayerDeltas = new ArrayList<double[]>(outputs.size() - 1);
		double[] outputDeltas = new double[layers.get(layers.size() - 1).length];
		for (int i = 0; i < outputDeltas.length; i++) {
			double nextOutput = outputs.get(outputs.size() - 1)[i];
			outputDeltas[i] = nextOutput * (1 - nextOutput);
			if (i == index) {
				outputDeltas[i] *= (correctLabelTarget - nextOutput);
			} else {
				outputDeltas[i] *= ((1 - correctLabelTarget) - nextOutput);
			}
		}
		allLayerDeltas.add(outputDeltas);
		for (int i = 2; i < outputs.size(); i++) {
			double[] nextLayerDeltas = new double[layers.get(layers.size() - i).length];
			for (int j = 0; j < nextLayerDeltas.length; j++) {
				double sum = 0.0;
				NonLinearUnit[] followingLayer = layers.get(layers.size() - i + 1);
				double[] followingLayerDeltas = allLayerDeltas.get(i - 2);
				for (int k = 0; k < followingLayer.length; k++) {
					sum += followingLayer[k].getWeight(j) * followingLayerDeltas[k];
				}
				double nextOutput = outputs.get(outputs.size() - i)[j];
				nextLayerDeltas[j] = nextOutput * (1 - nextOutput) * sum;
			}
			allLayerDeltas.add(nextLayerDeltas);
		}
		updateWeights(outputs, allLayerDeltas);
	}
	
	private void updateWeights(List<double[]> outputs, List<double[]> deltas) {
		for (int i = 0; i < layers.size(); i++) {
			NonLinearUnit[] nextLayer = layers.get(i);
			for (int j = 0; j < nextLayer.length; j++) {
				double[] weights = nextLayer[j].getWeights();
				for (int k = 0; k < weights.length; k++) {
					weights[k] += learningRate * deltas.get(deltas.size() - i - 1)[j] * outputs.get(i)[k];
				}
			}
		}
	}
}

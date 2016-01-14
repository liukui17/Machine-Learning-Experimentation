package data;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class MNISTParser {
	public static final String TRAIN_IMAGES = "./src/data/mnist/train-images.idx3-ubyte";
	public static final String TRAIN_LABELS = "./src/data/mnist/train-labels.idx1-ubyte";
	public static final String TEST_IMAGES = "./src/data/mnist/t10k-images.idx3-ubyte";
	public static final String TEST_LABELS = "./src/data/mnist/t10k-labels.idx1-ubyte";
	
	public static List<int[][]> readImages(String file) {
		try {
			DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
			int magic = in.readInt();
			int numImages = in.readInt();
			int imageHeight = in.readInt();
			int imageWidth = in.readInt();
		/*	System.out.println("Header Information: " + magic +
							   "(magic number) " + numImages +
							   "(image count) " + imageHeight +
							   "(height) " + imageWidth +
							   "(width)"); */
			List<int[][]> images = new LinkedList<int[][]>();
			for (int i = 0; i < numImages; i++) {
				int[][] nextImage = new int[imageHeight][imageWidth];
				for (int j = 0; j < imageHeight; j++) {
					for (int k = 0; k < imageWidth; k++) {
						nextImage[j][k] = (int) in.readByte() & 0xFF;
					}
				}
				images.add(nextImage);
			}
			in.close();
			return images;
		} catch (FileNotFoundException fnfe) {
			fnfe.printStackTrace(System.err);
		} catch (IOException ioe) {
			ioe.printStackTrace(System.err);
		}
		return null;
	}
	
	public static int[] readLabels(String file) {
		try {
			DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
			int magic = in.readInt();
			int numLabels = in.readInt();
			int[] labels = new int[numLabels];
			for (int i = 0; i < numLabels; i++) {
				labels[i] = (int) in.readByte() & 0xFF;
			}
			in.close();
			return labels;
		} catch (FileNotFoundException fnfe) {
			fnfe.printStackTrace(System.err);
		} catch (IOException ioe) {
			ioe.printStackTrace(System.err);
		}
		return null;
	}
	
	public static List<Instance<Integer,Integer>> makeInstances(List<int[][]> images, int[] labels, int start, int limit) {
		List<Instance<Integer,Integer>> instances = new ArrayList<Instance<Integer,Integer>>(limit);
		Iterator<int[][]> iter = images.iterator();
		int i = start;
		while (iter.hasNext() && i < limit + start && i < images.size()) {
			int[][] nextImage = iter.next();
			instances.add(makeInstance(nextImage, labels[i]));
			i++;
		}
		return instances;
	}
	
	public static Instance<Integer,Integer> makeInstance(int[][] image, int label) {
		List<Integer> attributes = new ArrayList<Integer>(image.length * image[0].length);
		
		/*
		 * There isn't much need to include the elements along the border since
		 * in almost all images, those pixels are 0. Of course, we can only do
		 * this because we know something about the data. In general, it isn't
		 * a good idea to remove features because some might actually be very
		 * influential in determining the classification.
		 */
		for (int i = 1; i < image.length - 1; i++) {
			for (int j = 1; j < image[i].length - 1; j++) {
				if (image[i][j] != 0) {
					attributes.add(1);
				} else {
					attributes.add(0);
				}
			//	attributes.add(image[i][j]);
			}
		}
		return new Instance<Integer,Integer>(attributes, label);
	}
	
	public static List<NumericalInstance<Integer>> makeNumericalInstances(List<int[][]> images, int[] labels, int start, int limit) {
		List<NumericalInstance<Integer>> instances = new ArrayList<NumericalInstance<Integer>>(limit);
		Iterator<int[][]> iter = images.iterator();
		int i = start;
		while (iter.hasNext() && i < limit + start && i < images.size()) {
			int[][] nextImage = iter.next();
			instances.add(makeNumericalInstance(nextImage, labels[i]));
			i++;
		}
		return instances;
	}
	
	public static NumericalInstance<Integer> makeNumericalInstance(int[][] image, int label) {
		double[] attributes = new double[image.length * image[0].length];
		for (int i = 0; i< image.length; i++) {
			for (int j = 0; j < image[i].length; j++) {
				if (image[i][j] != 0) {
					attributes[i * image[i].length + j] = 1;
				} else {
					attributes[i * image[i].length + j] = 0;
				}
			}
		}
		return new NumericalInstance<Integer>(attributes, label);
	}
}

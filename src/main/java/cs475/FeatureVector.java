package cs475;

import java.io.Serializable;
import java.util.*;

public class FeatureVector implements Serializable {

	private Map<Integer, Double> features = new HashMap<Integer, Double>();

	private double bias = 0;

	public FeatureVector() { }

	public void put(int index, double value) {
		if (value == bias) {
			features.remove(index);
		} else {
			features.put(index, value - bias);
		}
	}
	
	public double get(int index) {
		Double result = features.get(index);
		if (result == null) return bias;
		return result + bias;
	}

	public Map<Integer, Double> getFeatures() {
		return features;
	}

	public void addVector(FeatureVector v2) {
		for (Map.Entry<Integer, Double> entry: v2.features.entrySet()) {
			Double current = features.get(entry.getKey());
			features.put(entry.getKey(), entry.getValue() + (current == null ? 0 : current));
		}
		this.bias += v2.getBias();
	}

	public void addBias(double bias) {
		this.bias = bias;
	}

	public double getBias() {
		return this.bias;
	}

	public void multiplyConstant(double c) {
		for (Map.Entry<Integer, Double> entry: features.entrySet()) {
			features.put(entry.getKey(), entry.getValue() * c);
		}
		this.bias *= c;
	}

	public double valueSum() {
		double sum = 0;
		for (Map.Entry<Integer, Double> entry: features.entrySet()) {
			sum += entry.getValue() + bias;
		}
		return sum;
	}

	public double logValueSum() {
		double sum = 0;
		for (Map.Entry<Integer, Double> entry: features.entrySet()) {
			sum += Math.log(entry.getValue() + bias);
		}
		return sum;
	}

	void debugDump() {
		System.out.println("Bias = " + bias);
		for (Map.Entry<Integer, Double> entry: features.entrySet()) {
			System.out.println("Entry #" + entry.getKey() + ": " + entry.getValue());
		}
	}

}

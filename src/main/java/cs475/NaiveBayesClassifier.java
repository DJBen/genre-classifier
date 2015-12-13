package cs475;

import java.util.*;

public class NaiveBayesClassifier extends Classifier {
	
	private List<String> labels;
	private List<Double> priors;
	private List<FeatureVector> likelihoods;
	private int vocabularySize;
	private static final double SMOOTH_CONSTANT = 1;

	public NaiveBayesClassifier(int vocabularySize) {
		this.vocabularySize = vocabularySize;
	}

	public void train(Map<String, List<FeatureVector>> songs) {
		// Calculate P(c_j) : # of songs with genre = j / total song count
		priors = new ArrayList<>(); 
		labels = new ArrayList<>();
		int totalGenreCount = 0;
		for (String label : songs.keySet()) {
		 	List<FeatureVector> instances = songs.get(label);
		 	priors.add((double)instances.size());
			totalGenreCount += instances.size();
		}
		for (int i = 0; i < priors.size(); i++) {
			priors.set(i, priors.get(i) * 1.0 / totalGenreCount);
		}

		// Calculate P(w | c_j) = (n_k + \alpha) / (n + \alpha * size of vocab)
		// where \alpha is the smooth constant, preventing the program from conditioning on 0
		// Word frequency per genre (n_k): # of word w_k in genre c_j
		List<FeatureVector> wordFrequencies = new ArrayList<>();
		// Total word frequency per genre (n): # of words in genre c_j
		List<Double> totalWordFrequencies = new ArrayList<>();
		for (String label : songs.keySet()) {
			FeatureVector wordFrequencyInGenre = new FeatureVector();
			double frequencyCount = 0;
		 	List<FeatureVector> instances = songs.get(label);
		 	for (FeatureVector instance : instances) {
		 		wordFrequencyInGenre.addVector(instance);
		 		frequencyCount += instance.valueSum();
		 	}
		 	labels.add(label);
		 	// Add alpha
		 	wordFrequencyInGenre.addBias(SMOOTH_CONSTANT);
		 	wordFrequencies.add(wordFrequencyInGenre);
		 	totalWordFrequencies.add(frequencyCount + SMOOTH_CONSTANT * vocabularySize);
		}

		likelihoods = new ArrayList<>();
		for (int i = 0; i < wordFrequencies.size(); i++) {
			FeatureVector freq = wordFrequencies.get(i);
			double totalWordFreq = totalWordFrequencies.get(i);
			freq.multiplyConstant(1.0 / totalWordFreq);
			likelihoods.add(freq);
		}
	}

	public String classify(FeatureVector songFeature) {
		int maxIndex = -1;
		double maxLogPosterior = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < likelihoods.size(); i++) {
			double prior = priors.get(i);
			FeatureVector likelihood = likelihoods.get(i);
			double logPosterior = Math.log(prior);
			for (Map.Entry<Integer, Double> entry : songFeature.getFeatures().entrySet()) {
				int index = entry.getKey();
				double frequency = entry.getValue();
				logPosterior += frequency * Math.log(likelihood.get(index));
			}
			if (logPosterior > maxLogPosterior) {
				maxLogPosterior = logPosterior;
				maxIndex = i;
			}
		}
		return labels.get(maxIndex);
	}
}
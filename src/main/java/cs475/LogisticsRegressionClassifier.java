package cs475;

import jsat.classifiers.ClassificationDataSet;
import jsat.io.LIBSVMLoader;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by DJBen on 12/14/15.
 */
public class LogisticsRegressionClassifier extends Classifier {

    private int iterations;
 	private double initialLearningRate;
 	private FeatureVector weights = new FeatureVector();
 	private FeatureVector ftjSquareSum = new FeatureVector();
	private String predicting;

 	// g
 	private double logit(double z) {
 		return 1.0 / (1 + Math.exp(-z));
 	}

	public LogisticsRegressionClassifier(String predictingLabel) {
		this(predictingLabel, 20, 0.1);
	}

 	public LogisticsRegressionClassifier(String predictingLabel, int iterations, double initialLearningRate) {
		this.predicting = predictingLabel;
		this.iterations = iterations;
 		this.initialLearningRate = initialLearningRate;
 	}

    @Override
    public void train(Map<String, List<FeatureVector>> songs) throws Exception {
		for (int currentIteration = 0; currentIteration < iterations; currentIteration++) {
			for (String label : songs.keySet()) {
				for (FeatureVector features : songs.get(label)) {
					// x_i
					// Create gradient ascent
					FeatureVector weightsAscent = new FeatureVector();
					double labelValue = label.equals(predicting) ? 1 : 0;

					for (Map.Entry<Integer, Double> entry: features.entries()) {
						int j = entry.getKey();
						double jthFeature = entry.getValue();
						double jthGradient = labelValue * this.logit(-weights.innerProduct(features)) * jthFeature + (1 - labelValue) * this.logit(weights.innerProduct(features)) * (-jthFeature);
						ftjSquareSum.put(j, ftjSquareSum.get(j) + Math.pow(jthGradient, 2));
						double jthEta = initialLearningRate / Math.sqrt(1 + ftjSquareSum.get(j));
						weightsAscent.put(j, jthGradient * jthEta);
					}

					weights.addVector(weightsAscent);
				}
			}
 		}
    }

    @Override
    public String classify(FeatureVector songFeature) {
 		double result = this.logit(weights.innerProduct(songFeature));
 		return result < 0.5 ? null : predicting;
    }

	public static void trainAndValidate(Map<String, List<FeatureVector>> songs, Map<String, List<FeatureVector>> testSongs) throws Exception {
		List<String> genres = Arrays.asList(0,1,2,3,4,5,6,7,8,9).stream().map((n) -> Classify.intToGenre(n)).collect(Collectors.toList());
		System.out.println("Binary logistics classifier on every genre:");
		System.out.println("\t\tPositiv\tFalsePos\tFalseNeg\tAccuracy");
		for (String genre : genres) {
			Classifier c = new LogisticsRegressionClassifier(genre);
			c.train(songs);
			int positive = 0;
			int total = 0;
			int falsePositive = 0;
			int falseNegative = 0;

			for (String label : songs.keySet()) {
				for (FeatureVector features : songs.get(label)) {
					String result = c.classify(features);
					if (result != null) {
						if (genre.equals(label)) {
							positive++;
						} else {
							falsePositive++;
						}
					} else {
						if (label.equals(genre)) {
							falseNegative++;
						}
					}
					total++;
				}
			}
			System.out.println(Classify.shortenGenre(genre) + "\t" + positive + "\t\t" + falsePositive + "\t\t\t" + falseNegative + "\t\t\t" + (positive * 1.0) / (positive + falseNegative + falsePositive));
		}
	}
}

package cs475;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

import jsat.classifiers.linear.StochasticMultinomialLogisticRegression;
import jsat.io.LIBSVMLoader;
import jsat.classifiers.*;

/**
 * Created by DJBen on 12/12/15.
 */
public class SoftmaxClassifier extends Classifier {

    private double learningRate = 0.1;
    private int iterations = 100;

    private jsat.classifiers.Classifier classifier;

    public SoftmaxClassifier() {
    }

    public SoftmaxClassifier(int iterations, double learningRate) {
        this.learningRate = learningRate;
        this.iterations = iterations;
    }

    @Override
    public void train(Map<String, List<FeatureVector>> songs) throws IOException {
        File file = new File("data/dataset_train.libsvm");
        ClassificationDataSet dataSet = LIBSVMLoader.loadC(file);
        classifier = new StochasticMultinomialLogisticRegression(learningRate, iterations);
        classifier.trainC(dataSet);
    }

    @Override
    public String classify(FeatureVector songFeature) {
        return null;
    }

    public void validate() throws IOException {
        File file = new File("data/dataset_test.libsvm");
        ClassificationDataSet dataSet = LIBSVMLoader.loadC(file);
        Map<Integer, Map<Integer, Integer>> matrix = new HashMap<>();
        for (int cat = 0; cat < 10; cat++) {
            Map<Integer, Integer> catMap = new HashMap<>();
            for (int subcat = 0; subcat < 10; subcat++) {
                catMap.put(subcat, 0);
            }
            matrix.put(cat, catMap);
        }
        for (int i = 0; i < dataSet.getSampleSize(); i++) {
            DataPoint point = dataSet.getDataPoint(i);
            int truth = dataSet.getDataPointCategory(i);
            CategoricalResults result = classifier.classify(point);
            int mostLikely = result.mostLikely();
            matrix.get(truth).put(mostLikely, matrix.get(truth).get(mostLikely) + 1);
        }
        Classify.printConfusionMatrix(matrix);
    }

}

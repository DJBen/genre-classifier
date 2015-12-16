package cs475;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

import jsat.classifiers.linear.StochasticMultinomialLogisticRegression;
import jsat.datatransform.PCA;
import jsat.io.LIBSVMLoader;
import jsat.classifiers.*;

/**
 * Created by DJBen on 12/12/15.
 */
public class SoftmaxClassifier extends Classifier {

    private double learningRate = 0.1;
    private int iterations = 20;
    private int maxDimensions = -1;
    private PCA transform;
    private jsat.classifiers.Classifier classifier;

    public SoftmaxClassifier() {
    }

    public SoftmaxClassifier(int maxDimensions) {
        this.maxDimensions = maxDimensions;
    }

    public SoftmaxClassifier(int iterations, double learningRate) {
        this.learningRate = learningRate;
        this.iterations = iterations;
    }

    public void train(String fileName) throws IOException {
        File file = new File(fileName);
        ClassificationDataSet dataSet = LIBSVMLoader.loadC(file);

        if (maxDimensions > 0) {
            System.out.println("Applying PCA transform " + maxDimensions);
            transform = new PCA(dataSet, maxDimensions);
            dataSet.applyTransform(transform);
            classifier = new StochasticMultinomialLogisticRegression(learningRate, iterations);
            classifier.trainC(dataSet);
        } else {
            classifier = new StochasticMultinomialLogisticRegression(learningRate, iterations);
            classifier.trainC(dataSet);
        }
    }

    @Override
    public void train(Map<String, List<FeatureVector>> songs) throws IOException {

    }

    @Override
    public String classify(FeatureVector songFeature) {
        return null;
    }

    public void validate(String fileName) throws IOException {
        File file = new File(fileName);
        ClassificationDataSet dataSet = LIBSVMLoader.loadC(file);
        if (maxDimensions > 0) {
            System.out.println("Applying PCA transform to test..." + maxDimensions);
            dataSet.applyTransform(transform);
        }
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

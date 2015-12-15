package cs475;

import java.io.*;
import java.util.*;
import java.util.stream.*;

public class Classify {

	private static Map<String, List<FeatureVector>> songs;
	private static Set<String> vocabulary;

	public static void main(String[] args) throws Exception {
		testVector();
		System.out.println("Naive bayes classifier:");
		loadDataSet("data/dataset_train.txt");
		Classifier classifier = new NaiveBayesClassifier(vocabulary.size());
		classifier.train(songs);
		testDataSet(classifier, "data/dataset_test.txt");
		System.out.println("\nSoftmax classifier:");
		SoftmaxClassifier softmax = new SoftmaxClassifier();
		softmax.train(null);
		softmax.validate();
		System.out.println("\nSoftmax classifier with PCA:");
		SoftmaxClassifier softmax2 = new SoftmaxClassifier(200);
		softmax2.train(null);
		softmax2.validate();
	}

	private static void testVector() {
		FeatureVector f1 = new FeatureVector();
		f1.put(1, 7);
		f1.put(3, 4);
		f1.put(8, 1);
		f1.addBias(1);
		MLAssert(f1.get(1), 8, "Test failed at #1");
		MLAssert(f1.get(2), 1, "Test failed at #2");
		MLAssert(f1.get(3), 5, "Test failed at #3");
		MLAssert(f1.get(8), 2, "Test failed at #4");
		MLAssert(f1.get(10), 1, "Test failed at #5");
		FeatureVector f2 = new FeatureVector();
		f2.put(1, 2);
		f2.put(4, 1);
		f2.addBias(2);
		f1.addVector(f2);
		MLAssert(f1.get(1), 7 + 1 + 2 + 2, "Test failed at #6");
		MLAssert(f1.get(4), 1 + 1 + 2, "Test failed at #7");
		MLAssert(f1.get(3), 4 + 1 + 2, "Test failed at #8");
		f1.multiplyConstant(0.5);
		MLAssert(f1.getBias(), 1.5, "Bias not right: " + f1.getBias());
		MLAssert(f1.get(1), (7 + 1 + 2 + 2) * 0.5, "Test failed at #9: " + f1.get(1) + ", should be " + (7 + 1 + 2 + 2) * 0.5);
		MLAssert(f1.get(3), (4 + 1 + 2) * 0.5, "Test failed at #10: " + f1.get(3) + ", should be " + (4 + 1 + 2) * 0.5);
		// f1.debugDump();
	}

	private static void MLAssert(double v1, double v2, String message) {
		if (v1 != v2) {
			System.out.println(message);
		}
	}

	private static void loadDataSet(String fileName) throws Exception {
		try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
    		String line;
    		songs = new HashMap<>();
    		vocabulary = new HashSet<>();
    		while ((line = br.readLine()) != null) {
    			line = line.trim();
    			if (line.charAt(0) == '#') {
    				continue;
    			}
    			if (line.charAt(0) == '%') {
    				String[] rawVocab = line.substring(1).split(",");
    				for (String word : rawVocab) {
    					vocabulary.add(word);
    				}
    				continue;
    			}
    			String[] lineComponents = line.split(",");
    			if (lineComponents.length == 0) continue;
    			String label = lineComponents[0];
    			FeatureVector features = new FeatureVector();
    			for (int i = 1; i < lineComponents.length; i++) {
    				String[] subcomponents = lineComponents[i].split(":");
    				int wordIndex = Integer.parseInt(subcomponents[0]);
    				int wordFreq = Integer.parseInt(subcomponents[1]);
    				features.put(wordIndex, wordFreq);
    			}
    			List<FeatureVector> list = songs.get(label);
    			if (list == null) {
    				list = new ArrayList<FeatureVector>();
    				songs.put(label, list);
    			}
    			list.add(features);
    		}
		}
	}

	private static void testDataSet(Classifier classifier, String fileName) throws Exception {
		try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
    		String line;
    		vocabulary = new HashSet<>();
    		Map<String, Map<String, Integer>> matrix = new HashMap<>();
    		Set<String> labels = songs.keySet();
    		for (String l1 : labels) {
    			Map<String, Integer> row = new HashMap<String, Integer>();
    			for (String l2 : labels) {
    				row.put(l2, 0);
    			}
    			matrix.put(l1, row);
    		}
    		while ((line = br.readLine()) != null) {
    			line = line.trim();
    			if (line.charAt(0) == '#' || line.charAt(0) == '%') {
    				continue;
    			}
    			String[] lineComponents = line.split(",");
    			if (lineComponents.length == 0) continue;
    			String label = lineComponents[0];
    			FeatureVector features = new FeatureVector();
    			for (int i = 1; i < lineComponents.length; i++) {
    				String[] subcomponents = lineComponents[i].split(":");
    				int wordIndex = Integer.parseInt(subcomponents[0]);
    				int wordFreq = Integer.parseInt(subcomponents[1]);
    				features.put(wordIndex, wordFreq);
    			}
    			String result = classifier.classify(features);
    			matrix.get(label).put(result, matrix.get(label).get(result) + 1);
    		}
    		Classify.printConfusionMatrix(matrix);
		}
	}

	public static String shortenGenre(String name) {
		switch (name) {
		case "dance and electronica":
			return "dance";
		case "jazz and blues":
			return "jazz";
		case "soul and reggae":
			return "soul";
		case "classic pop and rock":
			return "cls-p&r";
		case "classical":
			return "classic";
		default:
			return name;
		}
	}

	public static int genreToInt(String name) throws IllegalArgumentException {
        switch (name) {
            case "pop":
                return 0;
            case "dance and electronica":
                return 1;
            case "punk":
                return 2;
            case "jazz and blues":
                return 3;
            case "soul and reggae":
                return 4;
            case "folk":
                return 5;
            case "metal":
                return 6;
            case "classic pop and rock":
                return 8;
            case "classical":
                return 7;
            case "hip-hop":
                return 9;
            default:
                throw new IllegalArgumentException("Genre name not exist");
        }
    }

    public static String intToGenre(int genreIndex) {
        final String[] genres = {"pop", "dance and electronica", "punk",
                "jazz and blues", "soul and reggae",
                "folk", "metal", "classic pop and rock",
                "classical", "hip-hop"};
        return genres[genreIndex];
	}

    public static <T> void printConfusionMatrix(Map<T, Map<T, Integer>> matrix) {
        System.out.println("\t" + String.join("\t", matrix.keySet().stream().map( p ->
                (p instanceof String ? Classify.shortenGenre((String)p) : (p instanceof Integer ? Classify.shortenGenre(Classify.intToGenre((Integer)p)) : null))
        ).collect(Collectors.toList())) + "\taccuracy");
        for (T category : matrix.keySet()) {
            String label = (category instanceof String) ? (String)category : Classify.intToGenre((Integer)category);
            System.out.print(Classify.shortenGenre(label) + "\t");
            Map<T, Integer> row = matrix.get(category);
            int accurate = 0;
            int inaccurate = 0;
            for (T subcat : row.keySet()) {
                int freq = row.get(subcat);
                if (category.equals(subcat)) {
                    accurate += freq;
                } else {
                    inaccurate += freq;
                }
                System.out.print(freq + "\t");
            }
            System.out.printf("%.2g\n", accurate * 1.0 / (accurate + inaccurate));
        }
    }

}
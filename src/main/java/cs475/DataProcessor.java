package cs475;

import java.util.*;
import java.io.*;

public class DataProcessor {
	// track to genre mapping
	private static Map<String, String> trackGenres = new HashMap<>();

	private static String FILE_NAME = "data/msd_genre_dataset.txt";

	public static void main(String[] args) {
		try (BufferedReader br = new BufferedReader(new FileReader(FILE_NAME))) {
    		String line;
    		while ((line = br.readLine()) != null) {
    			line = line.trim();
    			if (line.charAt(0) == '#' || line.charAt(0) == '%') {
    				continue;
    			}
    			String[] features = line.split(",");
    			String genre = features[0];
    			String trackId = features[1];
    			trackGenres.put(trackId, genre);
    		}
			addGenreToDataset("data/mxm_dataset_train.txt", "data/dataset_train.txt");
			addGenreToDataset("data/mxm_dataset_test.txt", "data/dataset_test.txt");
		} catch (Exception exception) {
			System.out.println(exception);
		}
	}

	private static void addGenreToDataset(String fileName, String outName) {
		try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
			try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outName), "utf-8"))) {
    			String line;
    			int found = 0, notFound = 0;
    			int total = 0;
    			Set<String> genres = new HashSet<String>();
    			while ((line = br.readLine()) != null) {
	    			line = line.trim();
	    			if (line.charAt(0) == '#' || line.charAt(0) == '%') {
	    				writer.write(line);
	    				writer.newLine();
	    				continue;
	    			}
	    			total++;
	    			String[] features = line.split(",");
	    			List<String> featureList = new LinkedList<String>(Arrays.asList(features));
	    			String trackId = features[0];
	    			String genre = trackGenres.get(trackId);
	    			if (genre != null) genres.add(genre);
	    			if (genre != null) {
	    				found++;
	    				featureList.remove(0);
	    				featureList.remove(0);
	    				// List<String> aggregatedFeatures = new LinkedList<String>();
	    				// for (String f : featureList) {
	    				// 	String[] map = f.split(":");
	    				// 	int freq = Integer.parseInt(map[1]);
	    				// 	if (freq >= 5) {
	    				// 		aggregatedFeatures.add(map[0] + ":" + "many");
	    				// 	} else if (freq >= 3) {
	    				// 		aggregatedFeatures.add(map[0] + ":" + "some");
	    				// 	} else {
	    				// 		aggregatedFeatures.add(map[0] + ":" + "few");
	    				// 	}
	    				// }
	    				writer.write(genre + ",");
	    				writer.write(String.join(",", featureList));
	    				writer.newLine();
	    			} else {
	    				notFound++;
	    			}
				}
				System.out.println("Converted: " + found + "; ignored: " + notFound + "; convertion rate: " + (found * 1.0 / total));
				System.out.println(genres.size() + " genres: " + genres);
				writer.close();
    		} catch (Exception e) {
    			System.out.println(e);
    		}
		} catch (Exception e) {
			System.out.println(e);
		}
	}
}
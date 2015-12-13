package cs475;

import java.io.*;
import java.util.*;

/**
 * Created by DJBen on 12/12/15.
 */
public class LIBSVMConverter {

    public static void main(String[] args) throws Exception {
        formalize("data/dataset_test.txt", "data/dataset_test.libsvm");
        formalize("data/dataset_train.txt", "data/dataset_train.libsvm");
    }

    public static void formalize(String inFile, String outFile) throws Exception {
        try (BufferedReader br = new BufferedReader(new FileReader(inFile))) {
            try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outFile), "utf-8"))) {
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.charAt(0) == '#') {
                        continue;
                    }
                    if (line.charAt(0) == '%') {
                        continue;
                    }
                    String[] features = line.split(",");
                    String genre = features[0];
                    int genreIndex = Classify.genreToInt(genre);
                    writer.write(genreIndex + " ");
                    for (int i = 1; i < features.length; i++) {
                        writer.write(features[i]);
                        if (i + 1 < features.length) {
                            writer.write(" ");
                        }
                    }
                    writer.newLine();
                }
            }
        }
    }

}

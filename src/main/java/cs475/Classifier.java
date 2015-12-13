package cs475;

import java.io.Serializable;
import java.util.*;

public abstract class Classifier implements Serializable {
	private static final long serialVersionUID = 1L;

	public abstract void train(Map<String, List<FeatureVector>> songs) throws Exception;

	public abstract String classify(FeatureVector songFeature);
}

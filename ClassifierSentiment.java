

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.FileUtils;

import com.google.common.collect.Lists;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class ClassifierSentiment {
    
    private Tokenizer tokenizer;
    private StopWordsRemoval stopWordsRemoval;
    private Classifier classifier;
    private Set<String> attributes;
    
    public ClassifierSentiment(String pathClassifier, String pathAttrs, Tokenizer tokenizer,StopWordsRemoval stopWordsRemoval) throws Exception {
        this.classifier = (Classifier) weka.core.SerializationHelper.read(pathClassifier);
        this.tokenizer = tokenizer;
        this.stopWordsRemoval = stopWordsRemoval;

        loadAttrs(pathAttrs);

    }
    
    private void loadAttrs(String pathName) throws IOException {

        this.attributes = new HashSet<>();
        for (String attr : FileUtils.readLines(new File(pathName))) {

            String[] data = attr.split(" ");
            // la clase es ignorada ya que los nuevos tweets no poseen clase
            if (data[1].equals("@@class@@")) {
                continue;
            }
            attributes.add(data[1]);
        }
    }
    
    public double classify(String content) throws Exception {
        
        List<String> tokenizedComment = tokenizer.getStrings(content);
        tokenizedComment = stopWordsRemoval.removeStopwords(tokenizedComment);

        InstanceComment comment = new InstanceComment(content, tokenizedComment, null);
        List<InstanceComment> comments = Lists.newArrayList();
        comments.add(comment);

		// Extracción de las características, si un tweet contiene la palabra se
        // agrega esta a un mapa attr --> value
        for (InstanceComment ins : comments) {
            
            for (String word : this.attributes) {
                
                if (ins.getTokenizedSource().contains(word)) {
                    ins.addFeature(word, 1);
                }
            }
        }
        // se transforman las instancias weka pero para el procedimiento de clasificacion.
        Instances instances = InstanceCommentUtils.toWekaInstancesForClassification(comments, this.attributes);
        
        return this.classifier.classifyInstance(instances.firstInstance());
    }
    
}

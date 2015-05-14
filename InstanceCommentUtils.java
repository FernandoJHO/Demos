
import java.util.List;
import java.util.Set;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class InstanceCommentUtils {
    
    private InstanceCommentUtils(){}
    
    //transforma la instancia de un comentario en una instancia de weka para el entrenamiento
    public static Instances toWekaInstances(List<InstanceComment> instancesComments, Set<String> attributes) {

        FastVector attrsWeka = new FastVector();

        for (String attr : attributes) {
            Attribute attribute = new Attribute(attr);
            attrsWeka.addElement(attribute);
        }

        FastVector classAttr = new FastVector();

        classAttr.addElement(CommentLabel.POSITIVO.name());
        classAttr.addElement(CommentLabel.NEGATIVO.name());

        Attribute classAttribute = new Attribute("@@class@@", classAttr);

        attrsWeka.addElement(classAttribute);

        Instances data = new Instances("Instances", attrsWeka, 0);

        data.setClassIndex(data.numAttributes() - 1);
        for (InstanceComment instanceComment : instancesComments) {
            Instance insWeka = new weka.core.Instance(attrsWeka.size()); //instance -> denseinstance
            insWeka.setDataset(data);
            for (Object obj : attrsWeka.toArray()) {
                Attribute attr = (Attribute) obj;
                if (attr.name().equals("@@class@@")) {
                    continue;
                }
                if (instanceComment.getFeature(attr.name()) == null) {
                    insWeka.setValue(attr, 0);
                } else {
                    insWeka.setValue(attr, instanceComment.getFeature(attr.name()));
                }
            }
            insWeka.setValue(classAttribute, instanceComment.getLabel().name());
            data.add(insWeka);

        }
        return data;

    }
    
    //transforma la instancia de un comentario en una instancia de weka para la clasificación
    public static Instances toWekaInstancesForClassification(List<InstanceComment> instancesComments, Set<String> attributes) {

        FastVector attrsWeka = new FastVector();

        for (String attr : attributes) {
            Attribute attribute = new Attribute(attr);
            attrsWeka.addElement(attribute);
        }

        FastVector classAttr = new FastVector();

        classAttr.addElement(CommentLabel.POSITIVO.name());
        classAttr.addElement(CommentLabel.NEGATIVO.name());

        Attribute classAttribute = new Attribute("@@class@@", classAttr);

        attrsWeka.addElement(classAttribute);

        Instances data = new Instances("Instances", attrsWeka, 0);

        data.setClassIndex(data.numAttributes() - 1);
        for (InstanceComment instanceComment : instancesComments) {
            Instance insWeka = new weka.core.Instance(attrsWeka.size()) {}; 
            insWeka.setDataset(data);
            for (Object obj : attrsWeka.toArray()) {
                Attribute attr = (Attribute) obj;
                if (attr.name().equals("@@class@@")) {
                    continue;
                }
                if (instanceComment.getFeature(attr.name()) == null) {
                    insWeka.setValue(attr, 0);
                } else {
                    insWeka.setValue(attr, instanceComment.getFeature(attr.name()));
                }
            }
            
            data.add(insWeka);

        }
        return data;

    }
    
}

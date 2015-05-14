
import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

import org.apache.commons.io.FileUtils;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

import com.google.common.collect.Lists;



public class TrainModelForSerialize {
    
    public TrainModelForSerialize(){}  
    
    //se entrena el modelo y genera la serialización de este para posteriores tareas de clasficación
    public void train() throws Exception{
        
        List<String> commentPositivos = FileUtils.readLines(new File("C:/Users/Fernando/Desktop/weka/positivo2.txt"));
        List<String> commentNegativos = FileUtils.readLines(new File("C:/Users/Fernando/Desktop/weka/negativo2.txt"));
        
        List<InstanceComment> comments = Lists.newArrayList();	
        
        Tokenizer tokenizer = new SimpleTokenizer(new Locale("ES"));
        StopWordsRemoval stopWordsRemoval = new StopWordsRemoval();
        
        //se procesan los comentarios positivos
        for (String comment : commentPositivos) {
            List<String> tokenizedComment = tokenizer.getStrings(comment);
            tokenizedComment = stopWordsRemoval.removeStopwords(tokenizedComment);
            InstanceComment instanceComment = new InstanceComment(comment, tokenizedComment, CommentLabel.POSITIVO);
            comments.add(instanceComment);
        }
        
        //se procesan los comentarios negativos
        for (String comment : commentNegativos) {
            List<String> tokenizedComment = tokenizer.getStrings(comment);
            tokenizedComment = stopWordsRemoval.removeStopwords(tokenizedComment);
            InstanceComment instanceComment = new InstanceComment(comment, tokenizedComment, CommentLabel.NEGATIVO);
            comments.add(instanceComment);
        }
        

        // Obtención de los atributos, que son las palabras presentes en un comentario
        Set<String> vocabulary = new HashSet<>();
        for (InstanceComment ins : comments) {
            vocabulary.addAll(ins.getTokenizedSource());
        }
        // si un comentario contiene una palabra contenida en el vocabulario, se marca la palabra
        for (InstanceComment ins : comments) {
            for (String word : vocabulary) {
                if (ins.getTokenizedSource().contains(word)) {
                    ins.addFeature(word, 1);
                }
            }
        }
        
        
        Instances data = InstanceCommentUtils.toWekaInstances(comments, vocabulary);



        // Se crea un clasificador SVM
        LibSVM classifier = new LibSVM();
        classifier.setGamma(0.1);
        
        //entrenamiento con el total de comentarios.
        classifier.buildClassifier(data);
        
        Evaluation modeleval = new Evaluation(data);
        
        modeleval.evaluateModel(classifier, data);


        //creación del modelo serializado
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("C:/Users/Fernando/Desktop/weka/svm.model"));
        oos.writeObject(classifier);
        oos.flush();
        oos.close();
        
        //se guardan los atributos usados
        StringBuilder attrs = new StringBuilder();
        for (int i = 0; i < data.numAttributes(); i++) {

            attrs.append(data.attribute(i)).append("\n");

        }
        
        //se genera un archivo con los atributos
        FileUtils.write(new File("C:/Users/Fernando/Desktop/weka/attrs.dat"), attrs.toString());
        

        
    }
    
}

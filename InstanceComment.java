

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

//clase que representa la instancia de un comentario

public class InstanceComment {
    
    private final String source;
    private final List<String> tokenizedSource;
    private final CommentLabel label;
    private Map<String, Integer> features;
    
    //constructor de la clase que recibe como parámetros el comentario original, el comentario tokenizado y la etiqueta del comentario
    public InstanceComment(String source, List<String> tokenizedSource,
            CommentLabel label) {
        super();
        this.source = source;
        this.tokenizedSource = tokenizedSource;
        this.label = label;
        this.features = new HashMap<>();
    }

    public String getSource() {
        return source;
    }

    public List<String> getTokenizedSource() {
        return tokenizedSource;
    }

    public Integer getFeature(String attr) {
        return this.features.get(attr);
    }

    public void addFeature(String attr, Integer value) {
        this.features.put(attr, value);
    }
    
    public CommentLabel getLabel() {
		return label;
    }
    
}

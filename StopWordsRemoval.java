

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import org.apache.commons.io.FileUtils;
import com.google.common.collect.Sets;

//clase que implementa métodos para eliminar pronombres, artículos o palabras comunes presentes en un comentario

public class StopWordsRemoval {
    
    private final Set<String> stopwords;

    public StopWordsRemoval() throws IOException {
        super();
        this.stopwords = loadStopwords();
    }

    private Set<String> loadStopwords() throws IOException {
        return Sets.newHashSet(FileUtils.readLines(new File("C:/Users/Fernando/Desktop/weka/stopwords-es.txt")));
    }

    //elimina las stopwords dado un comentario tokenizado
    public List<String> removeStopwords(List<String> words) {

        words.removeAll(this.stopwords);
        return words;

    }   
    
}


import java.util.List;
import java.util.Locale;

public interface Tokenizer {
    
    List<String> getStrings(String text);
    
    Locale getLocale();
    
}

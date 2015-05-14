
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class SimpleTokenizer implements Tokenizer {

    private Locale locale; //la clase Locale permite definir un idioma, en este caso será el español

    public SimpleTokenizer(Locale locale) {
        this.locale = locale;
    }

    //se tokeniza el comentario y se retorna una lista con las palabras que contiene
    @Override
    public List<String> getStrings(String text) {
        BreakIterator wordIterator = BreakIterator.getWordInstance(locale);
        wordIterator.setText(text);
        int start = wordIterator.first();
        int end = wordIterator.next();
        List<String> strings = new ArrayList<>();
        while (end != BreakIterator.DONE) {
            String word = text.substring(start, end);
            if (Character.isLetterOrDigit(word.charAt(0))) {
                strings.add(word.toLowerCase());
            }
            start = end;
            end = wordIterator.next();
        }
        return strings;
    }

    @Override
    public Locale getLocale() {
        return this.locale;
    }

}

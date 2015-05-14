
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances; 
 
public class Clima {
        //función que lee el archivo arff
	public static BufferedReader leerArchivo(String nombreArchivo) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(nombreArchivo));
		} catch (FileNotFoundException fnfe) {
			System.err.println("Archivo no encontrado: " + nombreArchivo);
		}
 
		return inputReader;
	}
        //Función que construye el clasificador, mediante un conjunto de datos de prueba y de entrenamiento.
	public static Evaluation clasificar(Classifier model,Instances datosEntreno, Instances datosPrueba) throws Exception {
		Evaluation evaluation = new Evaluation(datosEntreno);
 
		model.buildClassifier(datosEntreno);
		evaluation.evaluateModel(model, datosPrueba);
 
		return evaluation;
	}
 
        //Función que calcula la precisión de cada modelo construido.
	public static double calcularPrecision(FastVector predictions) {
		double correct = 0;
 
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
 
		return 100 * correct / predictions.size();
	}
 
        //Validación cruzada para entrenamiento.
	public static Instances[][] divisionValidacionCruzada(Instances data, int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
 
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
 
		return split;
	}
 
	public static void main(String[] args) throws Exception {
		BufferedReader datafile = leerArchivo("C:/Users/Fernando/Desktop/Weather.txt");
 
		Instances data = new Instances(datafile);
                //Establece el índice de la clase en el último atributo del archivo leído.
		data.setClassIndex(data.numAttributes() - 1);
 
		//10 iteraciones de validación cruzada
		Instances[][] split = divisionValidacionCruzada(data, 10);
 
		//Se separan o dividen los datos en dos, datos de entrenamiento y de prueba.
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
 
		//Se utilizan los algoritmos que contiene el arreglo models.
		Classifier[] models = { 
				new J48(), //Árbol de decisión.
				new PART(), //Usa división y conquista
				new DecisionTable(),//Tabla de decisión.
				new DecisionStump() //Árbol de decisión de un nivel.
		};
 
		//Se ejecuta para cada uno de los 4 modelos.
		for (int j = 0; j < models.length; j++) {
 
			//Todas las predicciones se guardan en un vector.
			FastVector predictions = new FastVector();
 
			//Se entrena y evalúa.
			for (int i = 0; i < trainingSplits.length; i++) {
				Evaluation validation = clasificar(models[j], trainingSplits[i], testingSplits[i]);
 
				predictions.appendElements(validation.predictions());
			}
 
			//Se calcula la precisión de cada modelo.
			double accuracy = calcularPrecision(predictions);
 
			//Imprime la precisión de cada modelo.
			System.out.println("Precisión de " + models[j].getClass().getSimpleName() + ": " + String.format("%.2f%%", accuracy) + "\n");
		}
 
        }
}

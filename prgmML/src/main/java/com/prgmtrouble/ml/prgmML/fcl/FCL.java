package com.prgmtrouble.ml.prgmML.fcl;

import java.io.Serializable;
import java.util.TreeSet;

import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;
import com.prgmtrouble.ml.prgmML.math.FunctionTypes;

/**
 * A Fully Connected Layer network consisting of 0 or more {@linkplain Layer}s and one
 * {@linkplain OutputLayer}. This class supports only the most basic functions of an FCL,
 * however can easily be extended to include other functions such as normalization.
 * 
 * @author prgmTrouble
 */
public class FCL implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	/**True if there are hidden layers.*/
	private final boolean deep;
	/**Array of hidden layers.*/
	private final Layer[] hidden;
	/**Output layer for this FCL.*/
	private final OutputLayer last;
	
	/**
	 * Creates a new FCL.
	 * 
	 * @param layers Hidden layers.
	 * @param output Output layer.
	 */
	public FCL(Layer[] layers, OutputLayer output) {
		deep = (layers != null);
		hidden = layers;
		last = output;
	}
	
	/**
	 * Creates a new FCL
	 * 
	 * @param inputSize Size of input vector.
	 * @param depth Number of layers.
	 * @param outputSize Size of output vector.
	 * @param functions Activation functions indexed by layer.
	 * @param parameters Input parameters for the forward and backward functions indexed
	 * 					 by layer. See {@linkplain FunctionTypes}, except set the first
	 * 					 value to null and initialize the hyperparameters (if applicable).
	 */
	public FCL(int inputSize, byte depth, int outputSize, FunctionTypes[] functions, Parameter<ListOfTypes>[][] parameters) {
		if(depth <= 0)
			error("Invalid depth (FCL cannot have "+depth+" layers).");
		if(functions == null)
			error("No functions provided.");
		if(functions.length != depth)
			error("Number of functions does not match depth.");
		
		int x = inputSize; 
		if(deep = (depth > 1)) { //If there should be hidden layers:
			x = Math.max(2, 2 * (inputSize + outputSize) / 3); //Optimal size for hidden layers.
			hidden = new Layer[depth - 1];
			
			for(byte i = 0; i < depth - 1; i++) { //For each hidden layer:
				final FunctionTypes f = functions[i]; //Create.
				hidden[i] = new Layer((f == null)? FunctionTypes.Blank:f, (i == 0)? inputSize:x, x, parameters[i]);
			}
		} else
			hidden = null;
		
		final FunctionTypes f = functions[depth - 1]; //Create output layer.
		last = new OutputLayer((f == null)? FunctionTypes.Blank:f, x, outputSize, parameters[depth - 1]);
	}
	
	/**
	 * Creates a new FCL
	 * 
	 * @param inputSize Size of input vector.
	 * @param depth Number of layers.
	 * @param outputSize Size of output vector.
	 * @param functions Activation functions indexed by layer.
	 * @param parameters Input parameters for the forward and backward functions indexed
	 * 					 by layer. See {@linkplain FunctionTypes}, except set the first
	 * 					 value to null and initialize the hyperparameters (if applicable).
	 * @param debug
	 */
	public FCL(int inputSize, byte depth, int outputSize, FunctionTypes[] functions, Parameter<ListOfTypes>[][] parameters, boolean debug) {
		if(depth <= 0)
			error("Invalid depth (FCL cannot have "+depth+" layers).");
		if(functions == null)
			error("No functions provided.");
		if(functions.length != depth)
			error("Number of functions does not match depth.");
		
		int x = inputSize; 
		if(deep = (depth > 1)) { //If there should be hidden layers:
			x = Math.max(2, 2 * (inputSize + outputSize) / 3); //Optimal size for hidden layers.
			hidden = new Layer[depth - 1];
			
			for(byte i = 0; i < depth - 1; i++) { //For each hidden layer:
				final FunctionTypes f = functions[i]; //Create.
				hidden[i] = new Layer((f == null)? FunctionTypes.Blank:f, (i == 0)? inputSize:x, x, parameters[i], debug);
			}
		} else
			hidden = null;
		
		final FunctionTypes f = functions[depth - 1]; //Create output layer.
		last = new OutputLayer((f == null)? FunctionTypes.Blank:f, x, outputSize, parameters[depth - 1], debug);
	}
	
	/**
	 * Sets the index of the expected output class as a parameter
	 * to the output function.
	 * 
	 * @param idx Index of expected class.
	 */
	public void setExpected(int idx) {last.setExpected(idx);}
	
	/**
	 * Performs the feed-forward operation.
	 * 
	 * @param in Input vector.
	 * @return The output of the FCL.
	 */
	public double[] forward(double[] in) {
		if(deep) //If there are hidden layers:
			for(Layer l : hidden) //For each hidden layer:
				in = (double[]) l.forward(in).getValues()[0]; //Perform feed-forward.
		return (double[]) last.forward(in).getValues()[0]; //Perform feed-forward on output layer.
	}
	
	/**
	 * Computes the gradient of the FCL with respect to the input vector as well as
	 * remove any layer elements which should be pruned.
	 * 
	 * @param learningRate Learning rate indexed by layer.
	 * @param prune Pruning threshold indexed by layer. Set to a negative value to disable.
	 * @param learnParameters Flags for each layer which indicate if the backpropagation
	 * 						  algorithm should edit their hyperparameters.
	 * @return The gradient with respect to the input vector.
	 */
	@SuppressWarnings("unchecked")
	public double[] backward(double[] learningRate, double[] prune, boolean[] learnParameters) {
		final boolean b = learnParameters == null; //True if parameter learning is specified.
		final int hl = (hidden != null)? hidden.length:0;
		
		Object[] bkwd = last.backward(learningRate[hl], prune[hl]); //Perform backpropagation on output layer.
		if(!b && learnParameters.length > hl && learnParameters[hl]) //If output layer should update hyperparameters:
			last.learnParameters(learningRate[hl]); //Update.
		
		for(int l = hl-1; l >= 0; l--) { //For each hidden layer starting from last:
			final Layer L = hidden[l];
			final double lr = learningRate[l];
			bkwd = L.backward((double[]) bkwd[0], lr, prune[l], (TreeSet<Integer>) bkwd[1]); //Perform backpropagation.
			if(!b && learnParameters.length > l && learnParameters[l]) //If output layer should update hyperparameters:
				L.learnParameters(lr); //Update.
		}
		
		return (double[]) bkwd[0]; //Return gradient with respect to input vector.
	}
	
	/**
	 * A custom exception which indicates an error in the FCL.
	 * 
	 * @author prgmTrouble
	 */
	private static class FCLException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "FCL Exception: ";
		
		public FCLException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws an {@linkplain FCLException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	protected static void error(String s) {
		try {
			throw new FCLException(s);
		} catch(FCLException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}




















































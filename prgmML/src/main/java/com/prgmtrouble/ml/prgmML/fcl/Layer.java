package com.prgmtrouble.ml.prgmML.fcl;

import java.io.Serializable;
import java.util.TreeSet;

import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;
import com.prgmtrouble.ml.prgmML.math.FunctionTypes;
import com.prgmtrouble.ml.prgmML.math.Vector;

/**
 * A {@linkplain WeightVector} whose destination is a {@linkplain Vector},
 * which typically represents a layer in a FCL network.
 * 
 * @author prgmTrouble
 */
public class Layer implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	/**The data for the layer.*/
	protected final Vector data;
	/**The weights connecting the input to the vector.*/
	private final WeightVector weights;
	/**The parameters used as inputs for functions.*/
	protected final Parameter<ListOfTypes>[] inParams;
	/**The parameters used as outputs for functions.*/
	private final Parameter<ListOfTypes>[] outParams;
	
	/**
	 * Creates a new layer.
	 * 
	 * @param activation Activation function. See {@linkplain FunctionTypes}.
	 * @param inSize Initial size of input.
	 * @param thisSize Initial size of this layer.
	 * @param inParameters Input parameters for the forward and backward functions.
	 * 					   See {@linkplain FunctionTypes}, except set the first value
	 * 					   to null and initialize the hyperparameters (if applicable).
	 */
	@SuppressWarnings("unchecked")
	public Layer(FunctionTypes activation, int inSize, int thisSize, Parameter<ListOfTypes>[] inParameters) {
		data = new Vector(activation);
		weights = new WeightVector(inSize, thisSize);
		inParams = inParameters;
		outParams = new Parameter[2];
	}
	
	/**
	 * Creates a new layer.
	 * 
	 * @param activation Activation function. See {@linkplain FunctionTypes}.
	 * @param inSize Initial size of input.
	 * @param thisSize Initial size of this layer.
	 * @param inParameters Input parameters for the forward and backward functions.
	 * 					   See {@linkplain FunctionTypes}, except set the first value
	 * 					   to null and initialize the hyperparameters (if applicable).
	 * @param debug
	 */
	@SuppressWarnings("unchecked")
	public Layer(FunctionTypes activation, int inSize, int thisSize, Parameter<ListOfTypes>[] inParameters, boolean debug) {
		data = new Vector(activation);
		weights = new WeightVector(inSize, thisSize, debug);
		inParams = inParameters;
		outParams = new Parameter[2];
	}
	
	/**
	 * Performs the feed-forward operation.
	 * 
	 * @param in Output from previous layer.
	 * @return The output of the activation function.
	 */
	public Parameter<ListOfTypes> forward(double[] in) {
		in = weights.forward(in); //Convert previous output to input.
		inParams[0].setValue(in,0); //Set input.
		return outParams[0] = data.forward(inParams[0]); //Get output.
	}
	
	/**
	 * Computes the gradient of the layer with respect to the inputs as well as
	 * remove any elements which should be pruned.
	 * 
	 * @param loss Incoming gradient from next layer.
	 * @param learningRate Learning rate.
	 * @param prune Pruning threshold. Set to a negative value to disable.
	 * @param toRemove A set of indices for elements which should be pruned (intended to be the
	 * 				   second output of the next layer's <code>backward</code> function).
	 * @return The gradient with respect to the previous layer's output as a <code>double[]</code>,
	 * 		   a <code>TreeSet</code> containing the indices of the previous layer which should be
	 * 		   removed, and the gradient with respect to the activation function (useful for learned
	 * 		   parameters) as a <code>Parameter</code>.
	 */
	@SuppressWarnings("unchecked")
	public Object[] backward(double[] loss, double learningRate, double prune, TreeSet<Integer> toRemove) {
		final boolean b = data.isOutputVector();
		if(!b)
			inParams[1].setValue(loss,0); //Set loss.
		Object[] wb;
		//Get gradient with respect to activation function, then get the gradient with respect to the previous outputs.
		final double[] nl = (double[]) (wb = weights.backward((double[]) (outParams[1] = data.backward(b? null:inParams[1])).getValues()[0], learningRate, prune))[0];
		TreeSet<Integer> src = null;
		if(prune >= 0.0) { //If pruning is enabled:
			src = (TreeSet<Integer>) wb[1]; //Get the set of indices to remove from the weights' backward function.
			if(toRemove == null)
				toRemove = new TreeSet<>();
			for(int i : src) //For each index:
				if(i < 0) { //If index is a destination:
					src.remove(i); //Remove index from source set.
					toRemove.add(-i); //Add inverted index to destination set.
				}
			weights.deleteDestination(toRemove); //Delete destinations.
			weights.deleteSource(src); //Delete sources.
		}
		wb = null;
		return new Object[] {nl,src,outParams[1]}; //Return the results.
	}
	
	@SuppressWarnings("unchecked")
	public Object[] backwardNoLearning(double[] loss, double learningRate, double prune, TreeSet<Integer> toRemove) {
		final boolean b = data.isOutputVector();
		if(!b)
			inParams[1].setValue(loss,0); //Set loss.
		Object[] wb;
		//Get gradient with respect to activation function, then get the gradient with respect to the previous outputs.
		final double[] nl = (double[]) (wb = weights.backwardNoLearning((double[]) (outParams[1] = data.backward(b? null:inParams[1])).getValues()[0], learningRate, prune))[0];
		TreeSet<Integer> src = null;
		if(prune >= 0.0) { //If pruning is enabled:
			src = (TreeSet<Integer>) wb[1]; //Get the set of indices to remove from the weights' backward function.
			if(toRemove == null)
				toRemove = new TreeSet<>();
			for(int i : src) //For each index:
				if(i < 0) { //If index is a destination:
					src.remove(i); //Remove index from source set.
					toRemove.add(-i); //Add inverted index to destination set.
				}
			weights.deleteDestination(toRemove); //Delete destinations.
			weights.deleteSource(src); //Delete sources.
		}
		wb = null;
		return new Object[] {nl,src,outParams[1]}; //Return the results.
	}
	
	/**
	 * Updates the hyperparameters (if any) according to the gradient
	 * calculated during backpropagation.
	 * 
	 * @param learningRate Learning rate.
	 */
	public void learnParameters(double learningRate) {data.learnParameters(learningRate);}
	
	/**
	 * A custom exception which indicates an error in the layer.
	 * 
	 * @author prgmTrouble
	 */
	private static class LayerException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "Layer Exception: ";
		
		public LayerException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws a {@linkplain LayerException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	protected static void error(String s) {
		try {
			throw new LayerException(s);
		} catch(LayerException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}


















































package com.prgmtrouble.ml.prgmML.fcl;

import java.util.TreeSet;

import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;
import com.prgmtrouble.ml.prgmML.math.FunctionTypes;

/**
 * An extension of the {@linkplain Layer} class that handles output functions.
 * 
 * @author prgmTrouble
 */
public class OutputLayer extends Layer {
	/***/
	private static final long serialVersionUID = 1L;
	
	/**
	 * Creates a new output layer.
	 * 
	 * @param activation An activation function which generates its own gradient.
	 * @param inSize Initial size of input.
	 * @param thisSize Initial size of this layer.
	 * @param parameters Input parameters for the forward and backward functions.
	 * 					 See {@linkplain FunctionTypes}, except set the first value
	 * 					 to null and initialize the others (if applicable).
	 */
	public OutputLayer(FunctionTypes activation, int inSize, int thisSize, Parameter<ListOfTypes>[] parameters) {
		super(activation, inSize, thisSize, parameters); //Basically the same constructor, except with the stipulation that the function
		if(!data.isOutputVector()) 						// must generate its own gradient.
			error("Output Layer: Invalid Function.");
	}
	
	/**
	 * Creates a new output layer.
	 * 
	 * @param activation An activation function which generates its own gradient.
	 * @param inSize Initial size of input.
	 * @param thisSize Initial size of this layer.
	 * @param parameters Input parameters for the forward and backward functions.
	 * 					 See {@linkplain FunctionTypes}, except set the first value
	 * 					 to null and initialize the others (if applicable).
	 * @param debug
	 */
	public OutputLayer(FunctionTypes activation, int inSize, int thisSize, Parameter<ListOfTypes>[] parameters, boolean debug) {
		super(activation, inSize, thisSize, parameters, debug); //Basically the same constructor, except with the stipulation that the function
		if(!data.isOutputVector()) 							   // must generate its own gradient.
			error("Output Layer: Invalid Function.");
	}
	
	/**
	 * Same as {@linkplain #backward(double, double)}.
	 * 
	 * @param loss Null.
	 * @param learningRate Learning rate.
	 * @param prune Pruning threshold. Set to a negative value to disable.
	 * @param toRemove Null.
	 */
	@Override
	public Object[] backward(double[] loss, double learningRate, double prune, TreeSet<Integer> toRemove) {return super.backward(null, learningRate, prune, null);}
	
	/**
	 * Computes the gradient of the layer with respect to the inputs as well as
	 * remove any elements which should be pruned.
	 * 
	 * @param learningRate Learning rate.
	 * @param prune Pruning threshold. Set to a negative value to disable.
	 * @return The gradient with respect to the previous layer's output as a <code>double[]</code>
	 * 		   and a <code>TreeSet</code> containing the indices of the previous layer which should
	 * 		   be removed.
	 */
	public Object[] backward(double learningRate, double prune) {return super.backward(null, learningRate, prune, null);}
	
	/**
	 * Sets the index of the expected output class as a parameter
	 * to the output function.
	 * 
	 * @param idx Index of expected class.
	 */
	public void setExpected(int idx) {inParams[0].setValue(idx,1);}
}























































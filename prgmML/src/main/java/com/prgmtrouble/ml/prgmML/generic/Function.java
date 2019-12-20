package com.prgmtrouble.ml.prgmML.generic;

import java.io.Serializable;

/**
 * A template for creating a function. A class which extends this
 * one is expected to provide both the function itself and its
 * derivative, both of which are responsible for setting the
 * appropriate parameter variables.
 * 
 * The {@linkplain Parameter} variables are essentially wrapper
 * variables for the inputs and outputs of both the forward and
 * backward functions. Their types are listed by their respective
 * {@linkplain ListOfTypes} variables in the constructor. 
 * 
 * @author prgmTrouble
 *
 * @param <T> An extension of the {@linkplain ListOfTypes} type.
 */
public abstract class Function<T extends ListOfTypes> implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	/**@return True if this function computes its own gradient.*/
	public abstract boolean isOutputFunction();
	
	/**Forward ListOfTypes.*/
	private final T fType;
	/**Backward ListOfTypes.*/
	private final T bType;
	/**Output ListOfTypes.*/
	private final T oType;
	/**Gradient ListOfTypes.*/
	private final T gType;
	/**Forward Parameters.*/
	private final Parameter<T> fParam;
	/**Backward Parameters.*/
	private final Parameter<T> bParam;
	/**Output Parameters.*/
	private final Parameter<T> oParam;
	/**Gradient Parameters.*/
	private final Parameter<T> gParam;
	
	/**
	 * Creates a new Function.
	 * 
	 * @param forwardType <code>ListOfTypes</code> for the forward parameters.
	 * @param backwardType <code>ListOfTypes</code> for the backward parameters.
	 * @param outputType <code>ListOfTypes</code> for the output parameters.
	 * @param gradientType <code>ListOfTypes</code> for the gradient parameters.
	 */
	public Function(T forwardType, T backwardType, T outputType, T gradientType) {
		fType = forwardType;
		bType = backwardType;
		oType = outputType;
		gType = gradientType;
		fParam = new Parameter<>(fType);
		bParam = new Parameter<>(bType);
		oParam = new Parameter<>(oType);
		gParam = new Parameter<>(gType);
	}
	
	/**
	 * Creates a new Function.
	 * 
	 * @param types An array of <code>ListOfTypes</code> objects that define the
	 * 				forward, backward, output, and gradient parameter types,
	 * 				respectively.
	 */
	@SuppressWarnings("unchecked")
	public Function(ListOfTypes[] types) {
		if(types == null || types.length != 4)
			throw new IllegalArgumentException("Invalid Type Set.");
		fType = (T) types[0];
		bType = (T) types[1];
		oType = (T) types[2];
		gType = (T) types[3];
		fParam = new Parameter<>(fType);
		bParam = new Parameter<>(bType);
		oParam = new Parameter<>(oType);
		gParam = new Parameter<>(gType);
	}
	
	/**
	 * Performs the forward operation for this function.
	 * 
	 * @param forwardParams Parameters for the forward operation.
	 * @return The output parameters.
	 */
	public abstract Parameter<T> forward(Object[] forwardParams);
	
	/**
	 * Performs the backward operation for this function.
	 * 
	 * @param backwardParams Parameters for the backward operation.
	 * @return The gradient parameters.
	 */
	public abstract Parameter<T> backward(Object[] backwardParams);
	
	/**
	 * Updates the hyperparameters (if any) according to the gradient
	 * calculated during backpropagation.
	 * 
	 * @param learningRate Learning rate.
	 */
	public abstract void learnParameters(double learningRate);
	
	/**
	 * Sets the forward parameters.
	 * 
	 * @param forwardParams Forward parameter values.
	 */
	public void setForwardParameter(Object[] forwardParams) {fParam.setValues(forwardParams);}
	/**
	 * Sets the backward parameters.
	 * 
	 * @param backwardParams Backward parameter values.
	 */
	public void setBackwardParameter(Object[] backwardParams) {bParam.setValues(backwardParams);}
	/**
	 * Sets the output parameters.
	 * 
	 * @param outputParams Output parameter values.
	 */
	public void setOutputParameter(Object[] outputParams) {oParam.setValues(outputParams);}
	/**
	 * Sets the gradient parameter values.
	 * 
	 * @param gradientParams Gradient parameter values.
	 */
	public void setGradientParameter(Object[] gradientParams) {gParam.setValues(gradientParams);}
	
	/**@return The input parameter for the feed-forward operation.*/
	public Parameter<T> getForwardParameter() {return fParam;}
	/**@return The input parameter for the backpropagation operation.*/
	public Parameter<T> getBackwardParameter() {return bParam;}
	/**@return The output parameter for the feed-forward operation.*/
	public Parameter<T> getOutputParameter() {return oParam;}
	/**@return The output parameter for the backpropagation operation.*/
	public Parameter<T> getGradientParameter() {return gParam;}
	
	/**
	 * Gets the types for each parameter.
	 * @return A {@linkplain ListOfTypes} array containing the types for the forward,
	 * 		   backward, output, and gradient parameters, respectively.
	 */
	@SuppressWarnings("unchecked")
	public T[] getTypes() {return (T[]) new ListOfTypes[] {fParam.getTypes(), bParam.getTypes(), oParam.getTypes(), gParam.getTypes()};}
	
	public abstract String toString();
}

















































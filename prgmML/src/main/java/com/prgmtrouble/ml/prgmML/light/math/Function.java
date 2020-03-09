package com.prgmtrouble.ml.prgmML.light.math;

import java.io.Serializable;

public class Function implements Serializable {
	
	/***/
	private static final long serialVersionUID = 1L;
	
	public static final double EXP_MIN_VALUE = Math.log(Double.MIN_VALUE);
	
	/**
	 * Types of functions.
	 */
	public static enum Type {
		ReLU,
		NSoftmax,
		CrossEntropy
	}
	
	/**Type of this function.*/
	private final Type type;
	/**<code>true</code> if function generates its own gradient.*/
	private final boolean isOutput;
	/**Output of function.*/
	private transient double[] out;
	/**Gradient of function.*/
	private transient double[] grad;
	
	/**
	 * Creates a new function object.
	 * 
	 * @param type Type of function.
	 */
	public Function(Type type) {
		isOutput = (this.type = type) == Type.CrossEntropy;
	}
	
	/**
	 * Fills a segment of an array with functions of the same type.
	 * If the input array is <code>null</code>, then a new array
	 * will be created with <code>end</code> elements.
	 * 
	 * @param arr Array.
	 * @param type {@linkplain Type} of function.
	 * @param start First index (inclusive).
	 * @param end Last index (exclusive).
	 * @return Array of functions.
	 */
	public static Function[] create(Function[] arr, Type type, int start, int end) {
		if(arr == null)
			arr = new Function[end];
		
		if(start >= end || start < 0 || end > arr.length)
			throw new IllegalArgumentException("Invalid bounds: start="+start+" end="+end+" arr.length="+arr.length);
		
		//Fill array.
		for(int i = start; i < end; i++)
			arr[i] = new Function(type);
		
		return arr;
	}
	
	/**
	 * Runs the activation function for non-output functions.
	 * 
	 * @param in Input.
	 */
	public void execute(double[] in) {
		switch(type) {
		case     ReLU:     ReLU(in); break;
		case NSoftmax: NSoftmax(in); break;
		case CrossEntropy:
			throw new IllegalArgumentException("Cross Entropy should be used with the `execute(double[],int)` function.");
		}
	}
	
	/**
	 * Runs the activation function for one-hot output functions.
	 * 
	 * @param in Input.
	 * @param exp Index of expected output.
	 */
	public void execute(double[] in, int exp) {
		switch(type) {
		case CrossEntropy: CrossEntropy(in, exp); break;
		case     ReLU:
		case NSoftmax: execute(in); break; //Protect user from itself.
		}
	}
	
	/**@return <code>true</code> if function generates its own gradient.*/
	public boolean isOutput() {return isOutput;}
	
	/**@return Output of activation function.*/
	public double[] getOut() {return out;}
	
	/**@return Gradient of activation function.*/
	public double[] getGrad() {return grad;}
	
	/**
	 * Rectified Linear Unit.
	 * 
	 * @param in Mutable copy of input.
	 */
	private void ReLU(double[] in) {
		out = in;
		final int il = out.length;
		grad = new double[il];
		for(int i = 0; i < il; i++) {
			final double v = out[i];
			out[i] = Math.max(0.0,v);
			grad[i] = v < 0.0? 0.0:1.0;
		}
	}
	
	/**
	 * Normalized Softmax function.
	 * 
	 * @param in Mutable copy of input.
	 */
	private void NSoftmax(double[] in) {
		out = in;
		final int il = out.length;
		double max = out[0];
		for(int i = 1; i < il; i++)
			if(out[i] > max)
				max = out[i];
		
		double eSum = 0.0;
		for(int i = 0; i < il; i++)
			eSum += out[i] = Math.exp(Math.max(out[i] - max,EXP_MIN_VALUE)); // In = I - max(I)
		
		grad = new double[il];
		
		for(int i = 0; i < il; i++) {
			final double v = out[i] /= eSum; // Oi = (b^(Ini)) / sum(b^(In))
			grad[i] = v * (1.0 - v);
		}
	}
	
	/**
	 * Cross Entropy one-hot output function (with normalized softmax).
	 * 
	 * @param in Mutable copy of input.
	 * @param exp Index of expected output.
	 */
	private void CrossEntropy(double[] in, int exp) {
		out = in;
		final int il = out.length;
		double max = out[0];
		for(int i = 1; i < il; i++)
			if(out[i] > max)
				max = out[i];
		
		double eSum = 0.0;
		for(int i = 0; i < il; i++)
			eSum += out[i] = Math.exp(Math.max(out[i] - max,EXP_MIN_VALUE)); // In = I - max(I)
		
		grad = new double[il];
		
		for(int i = 0; i < il; i++) {
			final double v = out[i] /= eSum; // Oi = (b^(Ini)) / sum(b^(In))
			grad[i] = v * (1.0 - v);
		}
		
		grad[exp] = -1.0;
	}
	
}















































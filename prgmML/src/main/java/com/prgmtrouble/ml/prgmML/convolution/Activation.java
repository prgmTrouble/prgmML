package com.prgmtrouble.ml.prgmML.convolution;

import java.io.Serializable;

import org.apache.commons.lang3.ArrayUtils;

import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;
import com.prgmtrouble.ml.prgmML.math.FunctionTypes;
import com.prgmtrouble.ml.prgmML.math.Tensor;
import com.prgmtrouble.ml.prgmML.math.Vector;

/**
 * A class similar to {@linkplain Vector} which takes a flattened input tensor, runs the
 * activation function over its entries, and finally runs a convolution operation.
 * 
 * This class does not support filters with different sizes.
 * 
 * @author prgmTrouble
 */
public class Activation implements Serializable,ConvolutionLayer {
	/***/
	private static final long serialVersionUID = 1L;
	
	private final Vector act;
	private Filter[] filters;
	private Parameter<ListOfTypes> out;
	private double lr = 0.1;
	private int[] fwd;
	private int[] bkwd;
	
	/**
	 * Creates a convolutional activation.
	 * 
	 * @param filters Convolutional filters.
	 * @param function Type of activation function.
	 */
	public Activation(Filter[] filters, FunctionTypes function) {act = new Vector(function); this.filters = filters;}
	
	/**
	 * Creates a convolutional activation with preset hyperparameters.
	 * 
	 * @param filters Convolutional filters.
	 * @param function Type of activation function.
	 * @param inputSize Input map side length.
	 * @param filterSize Filter side length.
	 * @param step Step size.
	 * @param channels Number of channels.
	 * @param pad Padding.
	 * @param lossSize Side length of gradient map.
	 * @param learningRate Learning rate.
	 */
	public Activation(Filter[] filters, FunctionTypes function,
					  int inputSize, int filterSize, int step, int channels, int pad,
					  int lossSize, double learningRate) {
		this(filters, function);
		setForwardHyperparams(inputSize, filterSize, step, channels, pad);
		setBackwardHyperparams(lossSize, learningRate);
	}
	
	/**
	 * Performs a convolution on an array of filters with constant dimensions.
	 * 
	 * @param in Flattened input map. 
	 * @param is Input map side length.
	 * @param filters Filters.
	 * @param fs Filter side length.
	 * @param step Step size.
	 * @param c Channels.
	 * @param pad Padding.
	 * @return The flattened output maps indexed by filter.
	 */
	private static double[][] convolve(double[] in, int is, Filter[] filters, int fs, int step, int c, int pad) {
		final int fl = filters.length;
		final double[][] out = new double[fl][];
		for(int i = 0; i < fl; i++) {
			final Filter f = filters[i];
			out[i] = Tensor.convolve(in, is, f.get(), fs, c, step, pad);
		}
		return out;
	}
	
	/**
	 * Performs the feed-forward operation with convolution.
	 * 
	 * This function executes the activation function for each element
	 * and runs the convolution operation with the given filters. The
	 * output is then flattened and stored in the same <code>Parameter</code>
	 * object output from the {@linkplain Vector#forward(Parameter)} function.
	 * 
	 * This function will run if the filters have different sizes, steps, and/or
	 * channels, however this function will not guarantee that the output is
	 * suitable as an input to another convolution layer. If possible, use
	 * the {@linkplain #setForwardHyperparams(int, int, int, int, int)} or
	 * {@linkplain #Activation(Filter[], FunctionTypes, int, int, int, int, int, int, double)}
	 * with {@linkplain #forward(Parameter)} functions instead.
	 * 
	 * @param input A {@linkplain Parameter} object containing the input
	 * 				(<code>double[]</code>) followed by any other parameters as 
	 * 			    required by the activation function.
	 * @param inputSize Input map side length.
	 * @param filters The filters for the convolution.
	 * @param filterSize Filter side length.
	 * @param step Step size.
	 * @param channels Number of channels.
	 * @param pad Padding.
	 * @return A {@linkplain Parameter} object containing the output of the
	 * 		   activation operation and a flattened output map from the
	 * 		   convolution.
	 */
	public Parameter<ListOfTypes> forward(Parameter<ListOfTypes> input, int inputSize, int filterSize, int step, int channels, int pad) {
		final Parameter<ListOfTypes> out = act.forward(input);
		double[][] nOut = convolve((double[]) out.getValues()[0], inputSize, filters, filterSize, step, channels, pad);
		double[] o = new double[0];
		for(double[] i : nOut)
			o = ArrayUtils.addAll(o,i);
		nOut = null;
		return this.out = new Parameter<ListOfTypes>(new ListOfTypes(act.getFunctionTypes()[0],new Class<?>[] {double[].class}),
										  			 ArrayUtils.add(out.getValues(), o));
	}
	
	/**
	 * Sets local storage for forward hyperparameters.
	 * 
	 * @param inputSize Input map side length.
	 * @param filterSize Filter side length.
	 * @param step Step size.
	 * @param channels Number of channels.
	 * @param pad Padding.
	 */
	public void setForwardHyperparams(int inputSize, int filterSize, int step, int channels, int pad) {fwd = new int[] {inputSize,filterSize,step,channels,pad};}
	
	/**
	 * Same as {@linkplain #forward(Parameter, int, int, int, int, int)},
	 * except uses the local values set by
	 * {@linkplain #setForwardHyperparams(int, int, int, int, int)} or
	 * {@linkplain #Activation(Filter[], FunctionTypes, int, int, int, int, int, int, double)}.
	 * 
	 * @param input A {@linkplain Parameter} object containing the input
	 * 				(<code>double[]</code>) followed by any other parameters as 
	 * 			    required by the activation function.
	 * @return A {@linkplain Parameter} object containing the output of the
	 * 		   activation operation and a flattened output map from the
	 * 		   convolution.
	 */
	@Override
	public Parameter<ListOfTypes> forward(Parameter<ListOfTypes> input) {
		if(fwd == null)
			error("setForwardHyperparams not called.");
		return forward(input, fwd[0], fwd[1], fwd[2], fwd[3], fwd[4]);
	}
	
	/**
	 * Performs the backpropagation operation with convolution.
	 * 
	 * This function assumes that the given parameters apply to all filters, where
	 * applicable. If possible, use {@linkplain #setForwardHyperparams(int, int, int, int, int)}
	 * and {@linkplain #setBackwardHyperparams(int)} or
	 * {@linkplain #Activation(Filter[], FunctionTypes, int, int, int, int, int, int, double)}
	 * with {@linkplain #backward(Parameter, double)} instead.
	 * 
	 * @param loss A {@linkplain Parameter} object containing the gradient with respect to
	 * 			   the output (<code>double[]</code>) followed by any other parameters as 
	 * 			   required by the activation function.
	 * @param learningRate Learning rate.
	 * @param inputSize Side length of input map.
	 * @param filterSize Side length of filter.
	 * @param lossSize Side length of gradient map.
	 * @param step Step size.
	 * @param channels Channels.
	 * @param pad Padding from forward convolution.
	 * @return Gradient with respect to the input map.
	 */
	public Parameter<ListOfTypes> backward(Parameter<ListOfTypes> loss, double learningRate, int inputSize, int filterSize, int lossSize, int step, int channels, int pad) {
		double[] nloss = (double[]) loss.getValues()[0];
		if(step > 1)
			nloss = Tensor.dilate(nloss, lossSize, channels, step - 1);
		final int nis = lossSize + (lossSize - 1) * (step - 1);
		double[] df = Tensor.scale(Tensor.convolve((double[]) out.getValues()[0], inputSize, nloss, nis, channels, 1, pad), learningRate);
		double[] di = new double[inputSize * inputSize * channels];
		for(Filter f : filters) {
			Tensor.sum(di, Tensor.convolve(nloss, nis, Tensor.rot180(f.get(), filterSize, channels), filterSize, channels, 1, filterSize));
			f.update(df);
		}
		nloss = null;
		loss.setValue(di,0);
		di = null;
		return act.backward(loss);
	}
	
	/**
	 * Sets the learning rate.
	 * 
	 * @param learningRate Learning rate.
	 */
	public void setLR(double learningRate) {lr = learningRate;}
	
	/**
	 * Sets local storage for backward hyperparameters.
	 * 
	 * @param lossSize Side length of gradient map.
	 * @param learningRate Learning rate.
	 */
	public void setBackwardHyperparams(int lossSize, double learningRate) {bkwd = new int[] {lossSize}; setLR(learningRate);}
	
	/**
	 * Same as {@linkplain #backward(Parameter, double, int, int, int, int, int, int)},
	 * except uses the local values set by
	 * {@linkplain #setForwardHyperparams(int, int, int, int, int)} and
	 * {@linkplain #setBackwardHyperparams(int)} or
	 * {@linkplain #Activation(Filter[], FunctionTypes, int, int, int, int, int, int, double)}.
	 * 
	 * @param loss A {@linkplain Parameter} object containing the gradient with respect to
	 * 			   the output (<code>double[]</code>) followed by any other parameters as 
	 * 			   required by the activation function.
	 * @return Gradient with respect to the input map.
	 */
	@Override
	public Parameter<ListOfTypes> backward(Parameter<ListOfTypes> loss) {
		if(fwd == null)
			error("setForwardHyperparams not called.");
		if(bkwd == null)
			error("setBackwardHyperparams not called.");
		return backward(loss, lr, fwd[0], fwd[1], bkwd[0], fwd[2], fwd[3], fwd[4]);
	}
	
	/**
	 * A custom exception which indicates an error in the activation map.
	 * 
	 * @author prgmTrouble
	 */
	private static class ActivationException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "Activation Exception: ";
		
		public ActivationException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws a {@linkplain ActivationException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	protected static void error(String s) {
		try {
			throw new ActivationException(s);
		} catch(ActivationException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}























































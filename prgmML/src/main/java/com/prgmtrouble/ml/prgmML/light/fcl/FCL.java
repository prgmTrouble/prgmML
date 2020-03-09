package com.prgmtrouble.ml.prgmML.light.fcl;

import java.io.Serializable;
import java.util.concurrent.ThreadLocalRandom;

import com.prgmtrouble.ml.prgmML.light.math.Function;
import com.prgmtrouble.ml.prgmML.math.Tensor;

public class FCL implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	private final double[][] weights;
	private transient double[] out;
	private final int[] size;
	private final int nl;
	private final boolean hasOutput;
	private final Function[] activations;
	
	/**
	 * Creates a new lightweight FCL.
	 * 
	 * @param activations An array of {@linkplain Function} objects, one for each layer.
	 * @param size An array containing the size of each layer.
	 */
	public FCL(Function[] activations, int[] size) {
		this.activations = activations;
		this.size = size;
		
		if(activations.length != size.length)
			throw new IllegalArgumentException("There must be at least one activation function per layer.");
		
		for(int i : size) //Ensure that sizes make sense.
			if(i < 1)
				throw new IllegalArgumentException("All layer sizes must be greater than zero.");
		
		nl = size.length - 1;
		hasOutput = activations[nl].isOutput();
		
		weights = new double[nl][];
		
		init();
	}
	
	/**
	 * Initializes the weights using Xavier initialization.
	 */
	private void init() {
		final ThreadLocalRandom r = ThreadLocalRandom.current();
		for(int i = 0; i < nl; i++) {
			final int nI = size[i],
					  nO = size[i+1],
					  ls = nI * nO;
			final double v = Math.sqrt(2.0 / (double) (nI + nO));
			final double[] layer = new double[ls];
			for(int j = 0; j < ls; j++)
				layer[j] = r.nextGaussian() * v;
			weights[i] = layer;
		}
	}
	
	/**
	 * Performs the main bulk of the feed-forward operation.
	 * 
	 * @param in Input to FCL.
	 */
	private void forwardHelper(double[] in) {
		out = in;
		
		for(int l = 0; l < nl; l++) { //For each layer:
			activations[l].execute(out); //Run activation function.
			//out = Tensor.dot(out,weights[l],size[l],size[l+1]); //Weight outputs for next layer.
			out = Tensor.dot(out,weights[l],1,size[l+1]);
		}
	}
	
	/**
	 * Performs the feed-forward operation for FCL networks without output functions.
	 * 
	 * @param in Input.
	 */
	public void forward(double[] in) {
		if(hasOutput)
			throw new IllegalArgumentException("FCLs with output functions must provide an expected value");
		forwardHelper(in);
		activations[nl].execute(out);
	}
	
	/**
	 * Performs the feed-forward operation for FCL networks with a one-hot output function.
	 * 
	 * @param in Input.
	 * @param exp Index of expected output.
	 */
	public void forward(double[] in, int exp) {
		forwardHelper(in);
		
		if(hasOutput)
			activations[nl].execute(out,exp);
		else
			activations[nl].execute(out); //Protect user from itself.
	}
	
	/**@return Final output of the network.*/
	public double[] getOut() {return out;}
	
	/**
	 * Performs the backpropagation operation for FCL networks with output functions.
	 * 
	 * @param lr Learning rate.
	 * @return The gradient w.r.t. the input of the FCL.
	 */
	public double[] backward(double lr) {		
		if(!hasOutput)
			throw new IllegalArgumentException("FCL functions without output layers must provide a gradient.");
		
		double[] gradient = activations[nl].getGrad();
		
		for(int l = nl - 1; l >= 0; l--) { //For each layer:
			final int nI = size[l],
					  nO = size[l+1],
					  ls = nI * nO;
			
			//Gradients w.r.t. layer input and weights, respectively.
			//final double[][] dl = Tensor.dDot(activations[l].getOut(), weights[l], gradient, nI, nO);
			final double[][] dl = Tensor.dDot(activations[l].getOut(), weights[l], gradient, 1, nO);
			
			//Pass gradient w.r.t. input to next iteration.
			gradient = Tensor.product(dl[0], activations[l].getGrad());
			
			//Update weights.
			final double[] layer = weights[l],
							  dw = dl[1];
			for(int i = 0; i < ls; i++)
				layer[i] -= lr * dw[i];
		}
		
		return gradient;
	}
	
	/**
	 * Performs the backpropagation operation for FCL networks without output functions.
	 * 
	 * @param gradient Gradient with respect to the FCL's output.
	 * @param lr Learning rate.
	 * @return The gradient w.r.t. the input of the FCL.
	 */
	public double[] backward(double[] gradient, double lr) {
		if(hasOutput) //Protect user from itself.
			return backward(lr);
		
		//Compute gradient w.r.t. input of final layer.
		Tensor.product(gradient, activations[nl].getGrad());
		
		for(int l = nl - 1; l >= 0; l--) { //For each layer:
			final int nI = size[l],
					  nO = size[l+1],
					  ls = nI * nO;
			
			//Gradients w.r.t. layer input and weights, respectively.
			final double[][] dl = Tensor.dDot(activations[l].getOut(), weights[l], gradient, nI, nO);
			
			//Pass gradient w.r.t. input to next iteration.
			gradient = Tensor.product(dl[0], activations[l].getGrad());
			
			//Update weights.
			final double[] layer = weights[l],
							  dw = dl[1];
			for(int i = 0; i < ls; i++)
				layer[i] -= lr * dw[i];
		}
		
		return gradient;
	}
	
	
}



















































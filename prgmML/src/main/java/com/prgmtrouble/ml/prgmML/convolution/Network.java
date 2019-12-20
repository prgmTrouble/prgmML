package com.prgmtrouble.ml.prgmML.convolution;

import java.io.Serializable;

import com.prgmtrouble.ml.prgmML.convolution.Pool.PoolingTypes;
import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;
import com.prgmtrouble.ml.prgmML.math.FunctionTypes;
import com.prgmtrouble.ml.prgmML.math.Vector;

public class Network implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	public static enum LayerTypes {
		Activation,
		Pool,
	}
	
	/**An array of {@linkplain ConvolutionLayer} objects.*/
	private final ConvolutionLayer[] network;
	/**A {@linkplain Vector} that handles the output of the function.*/
	private final Vector outputVector;
	/**True if {@linkplain #forward(Parameter)} function has been called.*/
	private boolean forwardExecuted = false;
	
	/**
	 * Creates a new convolutional network.
	 * 
	 * @param layers An array of {@linkplain ConvolutionLayer} objects.
	 * @param outputVector A {@linkplain Vector} with an output function.
	 * 
	 * @see Activation
	 * @see Pool
	 */
	public Network(ConvolutionLayer[] layers, Vector outputVector) {
		if(!outputVector.isOutputVector())
			error("Vector cannot be used as an output.");
		this.outputVector = outputVector;
		network = layers;
	}
	
	//TODO auto generate network
	public Network(int[] layerSizes, LayerTypes[] layerTypes, FunctionTypes[] activationTypes,
				   int[][] activationDimensions, PoolingTypes[] poolTypes, int[] poolingFactors,
				   int inputChannels, double learningRate) {
		final int nl = layerTypes.length-1; //Last element is output size.
		int aidx = 0,
			pidx = 0;
		network = new ConvolutionLayer[nl];
		for(int l = 0; l < nl; l++) {
			final int si = layerSizes[l],
					  so = layerSizes[l+1];
			switch(layerTypes[l]) {
			case Activation:
				{
					int[] fDim = activationDimensions[aidx];
					final int fs = fDim[0],
							  st = fDim[1],
							  ct = fDim[2];
					fDim = null;
					final Filter[] filters = new Filter[ct];
					for(int f = 0; f < ct; f++)
						filters[f] = new Filter(fs, inputChannels, st);
					network[l] = new Activation(filters, activationTypes[aidx++], si, fs, st, inputChannels, (so - 1) * st + fs - si, so, learningRate);
					inputChannels = ct;
				} break;
			}
		}
	}
	
	/**
	 * Performs the feed-forward operation across all layers in the network.
	 * 
	 * @param input Flattened input tensor (<code>double[]</code>) wrapped in
	 * 				a {@linkplain Parameter}.
	 * @return Flattened output vector (<code>double[]</code>) wrapped in
	 * 		   a {@linkplain Parameter}.
	 */
	public Parameter<ListOfTypes> forward(Parameter<ListOfTypes> input) {
		for(ConvolutionLayer layer : network)
			input = layer.forward(input);
		final Parameter<ListOfTypes> out = outputVector.forward(input);
		forwardExecuted = true;
		return out;
	}
	
	/**
	 * Performs the backpropagation operation across all layers in the network.
	 * 
	 * @return Gradient with respect to the input tensor.
	 */
	public Parameter<ListOfTypes> backward() {
		if(!forwardExecuted)
			error("Feed-forward function has not been called for the current cycle.");
		forwardExecuted = false;
		Parameter<ListOfTypes> loss = outputVector.backward(null);
		for(ConvolutionLayer layer : network)
			loss = layer.backward(loss);
		return loss;
	}
	
	/**
	 * A custom exception which indicates an error in the network.
	 * 
	 * @author prgmTrouble
	 */
	private static class NetworkException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "Network Exception: ";
		
		public NetworkException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws a {@linkplain NetworkException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	protected static void error(String s) {
		try {
			throw new NetworkException(s);
		} catch(NetworkException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}























































package com.prgmtrouble.ml.prgmML.convolution;

import java.io.Serializable;

import org.apache.commons.lang3.ArrayUtils;

import com.prgmtrouble.ml.prgmML.convolution.Pool.PoolingTypes;
import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;
import com.prgmtrouble.ml.prgmML.math.FunctionTypes;

/**
 * An object for managing convolution networks.
 * 
 * @author prgmTrouble
 */
public class Convolution implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	/**
	 * An enumeration of convolutional layer types.
	 * 
	 * @author prgmTrouble
	 */
	public static enum LayerTypes {
		/**@see Activation*/
		Activation,
		/**@see Pool*/
		Pool;
	}
	
	/**An array of {@linkplain ConvolutionLayer} objects.*/
	private final ConvolutionLayer[] network;
	/**{@linkplain #network}, but in reverse order.*/
	private final ConvolutionLayer[] reverseNetwork;
	/**True if {@linkplain #forward(Parameter)} function has been called.*/
	private boolean forwardExecuted = false;
	
	/**
	 * Creates a new convolutional network.
	 * 
	 * @param layers An array of {@linkplain ConvolutionLayer} objects.
	 * 
	 * @see Activation
	 * @see Pool
	 */
	public Convolution(ConvolutionLayer[] layers) {network = layers; ArrayUtils.reverse(reverseNetwork = ArrayUtils.clone(network));}
	
	/**
	 * Automatically generates a convolutional network with the parameters specified.
	 * Padding is automatically added between layers where necessary, though it is most
	 * effective when the output map size is less than or equal to the input map size.
	 * 
	 * @param layerSizes Side length of each input map, including the size of the output.
	 * @param layerTypes The type of each layer, indexed in order.
	 * @param activationTypes Type of activation function for each activation layer,
	 * 						  indexed in order.
	 * @param activationDimensions Filter size, step, and count for each activation
	 * 							   layer.
	 * @param poolTypes The type of pooling function for each pooling layer, indexed
	 * 					in order.
	 * @param poolingFactors The scaling factor for each pooling layer.
	 * @param inputChannels The number of channels in the input map.
	 * @param learningRate The initial learning rate for the network.
	 */
	public Convolution(int[] layerSizes, LayerTypes[] layerTypes, FunctionTypes[] activationTypes,
				       int[][] activationDimensions, PoolingTypes[] poolTypes, int[] poolingFactors,
				       int inputChannels, double learningRate) {
		final int nl = layerTypes.length, //Number of convolution layers.
				 rnl = nl - 1;
		int aidx = 0, //Activation layer counter.
			pidx = 0; //Pooling layer counter.
		network = new ConvolutionLayer[nl];
		reverseNetwork = new ConvolutionLayer[nl];
		for(int l = 0; l < nl; l++) { //For each layer:
			final int si = layerSizes[l], //Size of input.
					  so = layerSizes[l + 1]; //Size of output.
			switch(layerTypes[l]) {
			case Activation:
				{
					int[] fDim = activationDimensions[aidx]; //Dimensions of activation function.
					final int fs = fDim[0], //Filter size.
							  st = fDim[1], //Step size.
							  ct = fDim[2]; //Filter count.
					fDim = null;
					final Filter[] filters = new Filter[ct]; //Filters.
					for(int f = 0; f < ct; f++) //For each filter:
						filters[f] = new Filter(fs, inputChannels, st); //Create new.
					//Create new activation layer.
					network[l] = reverseNetwork[rnl - l] 
							= new Activation(filters, activationTypes[aidx++], si, fs, st, inputChannels, ((so - 1) * st + fs - si) / 2, so, learningRate);
					inputChannels = ct; //Update number of channels.
				} break;
			case Pool:
				{
					final PoolingTypes t = poolTypes[pidx]; //Pooling type.
					final int f = poolingFactors[pidx++]; //Pooling factor.
					switch (t) {
					case MaxC	:
					case MaxAbsC:
					case AvgC	:inputChannels /= f; break;
					default		:break;
					}
					network[l] = reverseNetwork[rnl - l] 
							= new Pool(si, inputChannels, f, t); //Create new pooling layer.
				} break;
			default: error("Invalid layer type."); break;
			}
		}
	}
	
	/**
	 * Performs the feed-forward operation across all layers in the network.
	 * 
	 * @param input Flattened input tensor (<code>double[]</code>) wrapped in
	 * 				a {@linkplain Parameter}.
	 * @return Flattened output vector (<code>double[]</code>) wrapped in
	 * 		   the input {@linkplain Parameter}.
	 */
	public Parameter<ListOfTypes> forward(Parameter<ListOfTypes> input) {
		for(ConvolutionLayer layer : network)
			input = layer.forward(input);
		forwardExecuted = true;
		return input;
	}
	
	/**
	 * Performs the backpropagation operation across all layers in the network.
	 * 
	 * @param loss A flattened gradient tensor (<code>double[]</code>) with
	 * 			   respect to the convolution output wrapped in a
	 * 			   {@linkplain Parameter}.
	 * @return Gradient with respect to the input tensor (<code>double[]</code>)
	 * 		   wrapped in the input {@linkplain Parameter}.
	 */
	public Parameter<ListOfTypes> backward(Parameter<ListOfTypes> loss) {
		if(!forwardExecuted)
			error("Feed-forward function has not been called for the current cycle.");
		forwardExecuted = false;
		for(ConvolutionLayer layer : reverseNetwork)
			loss = layer.backward(loss);
		return loss;
	}
	
	/**
	 * A custom exception which indicates an error in the convolutional network.
	 * 
	 * @author prgmTrouble
	 */
	private static class ConvolutionException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "Convolution Exception: ";
		
		public ConvolutionException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws a {@linkplain ConvolutionException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	protected static void error(String s) {
		try {
			throw new ConvolutionException(s);
		} catch(ConvolutionException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}























































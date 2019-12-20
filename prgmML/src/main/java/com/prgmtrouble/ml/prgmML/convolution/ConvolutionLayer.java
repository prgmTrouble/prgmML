package com.prgmtrouble.ml.prgmML.convolution;

import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;

/**
 * An interface for types of convolution layers.
 * 
 * @author prgmTrouble
 */
public interface ConvolutionLayer {
	
	/**
	 * Performs the feed-forward operation for the layer.
	 * @param input Flattened input tensor (<code>double[]</code>) wrapped in a {@linkplain Parameter}.
	 * @return Flattened output tensor (<code>double[]</code>) wrapped in a {@linkplain Parameter}.
	 */
	public Parameter<ListOfTypes> forward(Parameter<ListOfTypes> input);
	
	/**
	 * Performs the backpropagation operation for the layer.
	 * @param loss Gradient with respect to the layer output as a flattened tensor (<code>double[]</code>)
	 * 			   wrapped in a {@linkplain Parameter}.
	 * @return Gradient with respect to the layer input as a flattened tensor (<code>double[]</code>)
	 * 		   wrapped in a {@linkplain Parameter}.
	 */
	public Parameter<ListOfTypes> backward(Parameter<ListOfTypes> loss);
	
}

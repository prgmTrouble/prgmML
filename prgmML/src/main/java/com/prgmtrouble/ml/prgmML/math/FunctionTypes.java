package com.prgmtrouble.ml.prgmML.math;

/**
 * A list of function types.
 * 
 * @author prgmTrouble
 * 
 * @see #Blank
 * @see #ReLU
 * @see #LeakyReLU_DEF
 * @see #LeakyReLU
 * @see #ELU_DEF
 * @see #ELU
 * @see #ISRLU_DEF
 * @see #ISRLU
 * @see #SELU_DEF
 * @see #SELU
 * @see #SiLU_DEF
 * @see #SiLU
 * @see #Sigmoid_DEF
 * @see #Sigmoid
 * @see #TanH
 * @see #ArcTan
 * @see #ArcSinH
 * @see #Softmax_DEF
 * @see #Softmax
 * @see #NSoftmax_DEF
 * @see #NSoftmax
 * @see #CrossEntropy
 */
public enum FunctionTypes {
	/**
	 * Forward: In: null Out: null<br>
	 * Backward: In: null Out: null
	 */
	Blank,
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	ReLU,
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	LeakyReLU_DEF,
	/**
	 * Forward: In: double[],double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[],double[]
	 */
	LeakyReLU,
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	ELU_DEF,
	/**
	 * Forward: In: double[],double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[],double[]
	 */
	ELU,
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	ISRLU_DEF,
	/**
	 * Forward: In: double[],double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[],double[]
	 */
	ISRLU,
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	SELU_DEF,
	/**
	 * Forward: In: double[],double[],double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[],double[],double[]
	 */
	SELU,
	/**
	 * Forward: In: double[] Out: double[],double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	SiLU_DEF,
	/**
	 * Forward: In: double[],double[] Out: double[],double[]<br>
	 * Backward: In: double[] Out: double[],double[]
	 */
	SiLU,
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	Sigmoid_DEF,
	/**
	 * Forward: In: double[],double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[],double[]
	 */
	Sigmoid,
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	TanH,
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	ArcTan,
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	ArcSinH,
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	Softmax_DEF,
	/**
	 * Forward: In: double[],double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[],double[]
	 */
	Softmax,
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	NSoftmax_DEF,
	/**
	 * Forward: In: double[],double[] Out: double[],double[]<br>
	 * Backward: In: double[] Out: double[],double[]
	 */
	NSoftmax,
	/**
	 * Forward: In: double[],int Out: double[],double<br>
	 * Backward: In: null Out: double[]
	 */
	CrossEntropy;
}



















































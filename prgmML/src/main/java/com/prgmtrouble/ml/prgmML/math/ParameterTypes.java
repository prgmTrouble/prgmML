package com.prgmtrouble.ml.prgmML.math;

import com.prgmtrouble.ml.prgmML.generic.*;

/**
 * An enumeration of types for various functions. Each
 * item should contain a set for the input and output
 * for both forward and backward operations. 
 * 
 * @author prgmTrouble
 */
public enum ParameterTypes {
	/**
	 * Forward: In: null Out: null<br>
	 * Backward: In: null Out: null
	 */
	NULL(null),
	/**
	 * Forward: In: double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	ReLU(new ListOfTypes[] {new ListOfTypes(new Class<?>[] {double[].class}),
							new ListOfTypes(new Class<?>[] {double[].class}),
							new ListOfTypes(new Class<?>[] {double[].class}),
							new ListOfTypes(new Class<?>[] {double[].class})}),
	/**
	 * Forward: In: double[],double[] Out: double[]<br>
	 * Backward: In: double[] Out: double[],double[]
	 */
	LeakyReLU(new ListOfTypes[] {new ListOfTypes(new Class<?>[] {double[].class,double[].class}),
								 new ListOfTypes(new Class<?>[] {double[].class}),
								 new ListOfTypes(new Class<?>[] {double[].class}),
								 new ListOfTypes(new Class<?>[] {double[].class,double[].class})}),
	/**
	 * Forward: In: double[] Out: double[],double[]<br>
	 * Backward: In: double[] Out: double[]
	 */
	SELU(new ListOfTypes[] {new ListOfTypes(new Class<?>[] {double[].class,double[].class,double[].class}),
							new ListOfTypes(new Class<?>[] {double[].class}),
							new ListOfTypes(new Class<?>[] {double[].class}),
							new ListOfTypes(new Class<?>[] {double[].class,double[].class,double[].class})}),
	/**
	 * Forward: In: double[],double[] Out: double[],double[]<br>
	 * Backward: In: double[] Out: double[],double[]
	 */
	SiLU_DEF(new ListOfTypes[] {new ListOfTypes(new Class<?>[] {double[].class}),
								new ListOfTypes(new Class<?>[] {double[].class}),
								new ListOfTypes(new Class<?>[] {double[].class,double[].class}),
								new ListOfTypes(new Class<?>[] {double[].class})}),
	/**
	 * Forward: In: double[],double[] Out: double[],double[]<br>
	 * Backward: In: double[] Out: double[],double[]
	 */
	SiLU(new ListOfTypes[] {new ListOfTypes(new Class<?>[] {double[].class,double[].class}),
							new ListOfTypes(new Class<?>[] {double[].class}),
							new ListOfTypes(new Class<?>[] {double[].class,double[].class}),
							new ListOfTypes(new Class<?>[] {double[].class,double[].class})}),
	/**
	 * Forward: In: double[],int Out: double[],double<br>
	 * Backward: In: null Out: double[]
	 */
	CrossEntropy(new ListOfTypes[] {new ListOfTypes(new Class<?>[] {double[].class,Integer.class}),
									null,
									new ListOfTypes(new Class<?>[] {double[].class,Double.class}),
									new ListOfTypes(new Class<?>[] {double[].class})});
	
	/**The types for this parameter.*/
	private ListOfTypes[] types;
	
	/**
	 * Gets the types for this parameter.
	 * 
	 * @return An array of <code>ListOfTypes</code> objects.
	 */
	public ListOfTypes[] getTypes() {return types;}
	
	/**
	 * Creates a <code>ParameterTypes</code> enum element from an
	 * array of <code>ListOfTypes</code> objects.
	 * 
	 * @param types An array of <code>ListOfTypes</code> objects.
	 */
	private ParameterTypes(ListOfTypes[] types) {this.types = types;}
}

















































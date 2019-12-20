package com.prgmtrouble.ml.prgmML.generic;

import java.io.Serializable;

import org.apache.commons.lang3.ArrayUtils;

/**
 * A class that holds an array of Class objects to represent generic
 * types.
 * 
 * @author prgmTrouble
 */
public class ListOfTypes implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	/**An array of class objects representing types.*/
	private final Class<?>[] t;
	
	/**
	 * Create a list of types.
	 * 
	 * @param types An array of class objects representing types.
	 */
	public ListOfTypes(Class<?>[] types) {t = types;}
	
	/**
	 * Create a list of types.
	 * @param x A <code>ListOfTypes</code> object to copy types from.
	 * @param append An array of class objects representing types which will be appended.
	 */
	public ListOfTypes(ListOfTypes x, Class<?>[] append) {t = ArrayUtils.addAll(x.t,append);}
	
	/**
	 * Gets the classes.
	 * 
	 * @return The array of class objects representing types.
	 */
	public Class<?>[] getTypes() {return t;}
	
	/**
	 * Checks if the values can be cast to the types in this
	 * object.
	 * 
	 * @param values Values to be checked.
	 * @return True if all objects can be cast to their corresponding
	 * 		   types in this object.
	 */
	public boolean instanceOf(Object[] values) {
		final int tl = t.length;
		if(values.length != tl)
			return false;
		boolean b = true;
		for(int i = 0; i < t.length && (b = t[i].isInstance(values[i])); i++);
		return b;
	}
	
	/**
	 * Checks if the value can be cast to the type at the specified index.
	 * 
	 * @param value Value to be checked.
	 * @param idx Index of type.
	 * @return True if the value can be cast to the type at the specified index.
	 */
	public boolean instanceOf(Object value, int idx) {return t[idx].isInstance(value);}
}





















































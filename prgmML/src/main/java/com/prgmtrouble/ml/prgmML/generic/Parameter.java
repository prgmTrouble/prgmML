package com.prgmtrouble.ml.prgmML.generic;

import java.io.Serializable;

/**
 * A data structure which holds a set of immutable types and
 * allows objects of those types to be stored.
 * 
 * @author prgmTrouble
 *
 * @param <T> An extension of the {@linkplain ListOfTypes}.
 */
public class Parameter<T extends ListOfTypes> implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	/**A list of types for this parameter.*/
	private final T type;
	/**The values for this parameter.*/
	private Object[] val;
	
	/**
	 * Creates a new function parameter.
	 * 
	 * @param types A list of types for this parameter.
	 * @param values The objects themselves.
	 */
	public Parameter(T types, Object[] values) {type = types; setValues(values);}
	
	/**
	 * Creates a new function parameter with empty
	 * values.
	 * 
	 * @param types A list of types for this parameter.
	 */
	public Parameter(T types) {type = types;}
	
	/**@return The <code>ListOfTypes</code> object for this parameter.*/
	public T getTypes() {return type;}
	/**@return The objects held by this parameter as an <code>Object[]</code>.*/
	public Object[] getValues() {return val;}
	
	/**
	 * Sets the values of this parameter. 
	 * 
	 * @param values The new values.
	 */
	public void setValues(Object[] values) {
		if(!type.instanceOf(values))
			error("Values have incorrect type.");
		val = values;
	}
	/**
	 * Sets a value of the parameter.
	 * 
	 * @param value The new value.
	 * @param idx The index.
	 */
	public void setValue(Object value, int idx) {
		if(!type.instanceOf(value, idx))
			error("Value has incorrect type for index "+idx+".");
		if(val == null)
			val = new Object[idx + 1];
		else if(val.length <= idx) {
			final Object[] t = new Object[idx + 1];
			int i = 0;
			for(Object o : val)
				t[i++] = o;
			val = t;
		}
		val[idx] = value;
	}
	
	/**
	 * A custom exception which indicates an error in a parameter.
	 * 
	 * @author prgmTrouble
	 */
	private static class ParameterException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "Parameter Exception: ";
		
		public ParameterException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws a {@linkplain ParameterException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	private void error(String s) {
		try {
			throw new ParameterException(s);
		} catch (ParameterException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}


















































package com.prgmtrouble.ml.prgmML.convolution;

import java.io.Serializable;
import java.util.concurrent.ThreadLocalRandom;

/**
 * An object which holds the data for a flattened
 * convolutional filter.
 * 
 * @author prgmTrouble
 */
public class Filter implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	/**Data for this filter.*/
	private final double[] data;
	/**Side length.*/
	private final int s;
	/**Channels.*/
	private final int c;
	/**Step size.*/
	private final int step;
	
	/**
	 * Creates a new filter.
	 * 
	 * @param size Side length of filter.
	 * @param channels Channels.
	 * @param step Step size.
	 */
	public Filter(int size, int channels, int step) {
		s = size;
		c = channels;
		this.step = step;
		final int l = s * s * c;
		data = new double[l];
		final double x = Math.sqrt(2.0 / (double) l);
		ThreadLocalRandom r = ThreadLocalRandom.current();
		for(int i = 0; i < l; i++)
			data[i] = r.nextGaussian() * x;
	}
	
	/**
	 * Gets a specific value from the filter.
	 * 
	 * @param row Row index.
	 * @param col Column index.
	 * @param channel Channel index.
	 * @return Value at <code>[row][col][channel]</code>.
	 */
	public double get(int row, int col, int channel) {return data[(channel * s * s) + (row * s) + col];}
	
	/** @return The array for this filter.*/
	public double[] get() {return data;}
	
	/**@return Side length of filter.*/
	public int size() {return s;}
	/**@return Number of channels in filter.*/
	public int channels() {return c;}
	/**@return Step size.*/
	public int step() {return step;}
	
	/**
	 * Updates the filter.
	 * 
	 * @param gradient Gradient with respect to this filter, scaled by learning rate.
	 */
	public void update(double[] gradient) {
		for(int i = 0; i < Math.min(s,gradient.length); i++)
			data[i] -= gradient[i];
	}
}






















































package com.prgmtrouble.ml.prgmML.math;

public final class Tensor {
	/**
	 * Dilates a flattened rank-3 tensor.
	 * 
	 * @param in Input.
	 * @param is Input side length.
	 * @param c Channels.
	 * @param factor Number of empty rows and columns to add between elements.
	 * @return Flattened and dilated matrix.
	 */
	public static double[] dilate(double[] in, int is, int c, int factor) {
		final int ns  = is * (++factor) - 1, //Dilated side length.
				  nsq = ns * ns, //Dilated side length squared.
				  isq = is * is; //Input side length squared.
		final double[] out = new double[c * nsq]; //Output tensor.
		for(int ch = 0; ch < c; ch++) { //For each channel:
			final int chisq = ch * isq, //Input channel index.
					  chnsq = ch * nsq; //Dilated channel index.
			for(int ir = 0; ir < is; ir++) { //For each input row:
				final int xr = chisq + (ir * is), //Input row index.
						  nr = chnsq + (ir * factor * ns); //Dilated row index.
				for(int ic = 0; ic < is; ic++) //For each input column:
					out[nr + (ic * factor)] = in[xr + ic]; //Record input.
			}
		}
		return out; //Return dilated tensor.
	}
	
	public static double[] transpose(double[] in, int is, int c) {
		final int isq = is * is;
		for(int ch = 0; ch < c; ch++) {
			final int chq = ch * isq;
			for(int ir = 0; ir < is; ir++)
				for(int ic = 0; ic < is; ic++) {
					final int aidx = chq + (ir * is) + ic,
							  bidx = chq + (ic * is) + ir;
					final double t = in[aidx];
					in[aidx] = in[bidx];
					in[bidx] = t;
				}
		}
		return in;
	}
	
	public static double[] reverseColumns(double[] in, int is, int c) {
		final int isq = is * is;
		for(int ch = 0; ch < c; ch++) {
			final int chq = ch * isq;
			for(int ir = 0; ir < is; ir++)
				for(int ic = 0; ic < is; ic++) {
					final int aidx = chq + (ir * is) + ic,
							  bidx = chq + (ir * is) + (is - ic - 1);
					final double t = in[aidx];
					in[aidx] = in[bidx];
					in[bidx] = t;
				}
		}
		return in;
	}
	
	public static double[] rot180(double[] in, int is, int c) {
		final int il = in.length;
		final double[] out = new double[il];
		System.arraycopy(in, 0, out, 0, il);
		return reverseColumns(transpose(reverseColumns(transpose(out,is,c),is,c),is,c),is,c);
	}
	
	public static double[] sum(double[] a, double[] b) {
		int x = 0;
		for(double i : b)
			a[x++] += i;
		return a;
	}
	
	public static double[] scale(double[] a, double b) {
		for(int i = 0; i < a.length; i++)
			a[i] *= b;
		return a;
	}
	
	/**
	 * Performs a convolution operation using flattened arrays.
	 * 
	 * @param in Flattened input map.
	 * @param is Input map side length.
	 * @param filter Flattened filter.
	 * @param fs Filter side length.
 	 * @param c Channels.
	 * @param step Step size.
	 * @param pad Padding.
	 * @return The flattened output map.
	 */
	public static double[] convolve(double[] in, int is, double[] filter, int fs, int c, int step, int pad) {
		if((is + 2 * pad - fs) % step != 0) //TODO include pad in check?
			error("Input and filter size difference is not a factor of the step size.");
		
		final int os  = (is + 2 * pad - fs) / step,
				  osq = os * os,
				  isq = is * is,
				  fsq = fs * fs;
		final double[] out = new double[osq];
		
		for(int or = 0; or < os; or++) {
			final int oros = or * os,
					  orst = or * step;
			for(int oc = 0; oc < os; oc++) {
				final int ocst = oc * step;
				double o = 0.0;
				for(int ch = 0; ch < c; ch++) {
					final int chisq = ch * isq,
							  chfsq = ch * fsq;
					for(int fr = 0; fr < fs; fr++) {
						final int ir = orst + fr - pad,
								  frfs = fr * fs + chfsq;
						if(ir >= 0 && ir < is) {
							final int iris = ir * is + chisq;
							for(int fc = 0; fc < fs; fc++) {
								final int ic = ocst + fc - pad;
								if(ic >= 0 && ic < is)
									o += in[iris + ic]
									   * filter[frfs + fc];
							}
						}
					}
				}
				out[oros + oc] = o;
			}
		}
		return out;
	}
	
	/**
	 * A custom exception which indicates an error in a tensor.
	 * 
	 * @author prgmTrouble
	 */
	private static class TensorException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "Tensor Exception: ";
		
		public TensorException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws a {@linkplain TensorException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	protected static void error(String s) {
		try {
			throw new TensorException(s);
		} catch(TensorException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}

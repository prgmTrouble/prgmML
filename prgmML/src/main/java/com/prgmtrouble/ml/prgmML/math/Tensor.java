package com.prgmtrouble.ml.prgmML.math;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.concurrent.ThreadLocalRandom;

public final class Tensor {
	
	/**
	 * Creates a new tensor with an initialized value.
	 * 
	 * @param v Value.
	 * @param size Size.
	 * @return A new tensor with all values set to <code>v</code>.
	 */
	public static double[] newTensor(double v, int size) {
		final double[] out = new double[size];
		for(int i = 0; i < size; i++)
			out[i] = v;
		return out;
	}
	
	/**
	 * Creates a new vector with a gaussian distribution.
	 * 
	 * @param v Standard deviation.
	 * @param size Size of vector.
	 * @param r Random object.
	 * @return A vector with elements of a gaussian distribution.
	 */
	public static double[] gaussian(double v, int size, ThreadLocalRandom r) {
		final double[] out = new double[size];
		for(int i = 0; i < size; i++)
			out[i] = r.nextGaussian() * v;
		return out;
	}
	
	/**
	 * Duplicates a tensor.
	 * 
	 * @param in Input tensor.
	 * @return A copy of <code>in</code>.
	 */
	public static double[] dupe(double[] in) {
		final int il = in.length;
		final double[] out = new double[il];
		System.arraycopy(in, 0, out, 0, il);
		return out;
	}
	
	/**
	 * Dilates a flattened rank-3 square tensor.
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
	
	/**
	 * Transposes a flattened rank-3 square tensor in-place.
	 * 
	 * @param in Input tensor.
	 * @param is Input side length.
	 * @param c Input channel size.
	 * @return <code>(in)T</code>
	 */
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
	
	/**
	 * Transposes a flattened matrix.
	 * 
	 * @param in Input matrix.
	 * @param ir Input row length.
	 * @param ic Input column length.
	 * @return <code>(in)T</code>
	 */
	public static double[] transpose2(double[] in, int ir, int ic) {
		double[] out = new double[in.length];
		for(int r = 0; r < ir; r++) {
			final int ric = r * ic;
			for(int c = 0; c < ic; c++) {
				final int aidx = ric + c,
						  bidx = (c * ir) + r;
				out[bidx] = in[aidx];
			}
		}
		return out;
	}
	
	/**
	 * Reverses the columns of a flattened rank-3 square tensor in-place.
	 * 
	 * @param in Input tensor.
	 * @param is Input side length.
	 * @param c Input channel size.
	 * @return The input, but with the columns reversed.
	 */
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
	
	/**
	 * Rotates a flattened rank-3 square tensor 180°.
	 * 
	 * @param in Input tensor.
	 * @param is Input side length.
	 * @param c Input channel size.
	 * @return The input tensor, but rotated 180°.
	 */
	public static double[] rot180(double[] in, int is, int c) {
		final int il = in.length;
		final double[] out = new double[il];
		System.arraycopy(in, 0, out, 0, il);
		return reverseColumns(transpose(reverseColumns(transpose(out,is,c),is,c),is,c),is,c);
	}
	
	/**
	 * Adds the values of <code>b</code> to <code>a</code>.
	 * 
	 * @param a First tensor.
	 * @param b Second tensor.
	 * @return <code>a += b</code>.
	 */
	public static double[] sum(double[] a, double[] b) {
		if(a.length != b.length)
			error("Unequal tensor sizes.");
		int x = 0;
		for(double i : b)
			a[x++] += i;
		return a;
	}
	
	/**
	 * @param x
	 * @return The sum of all elements in <code>x</code>.
	 */
	public static double sum(double[] x) {
		double y = 0.0;
		for(double i : x)
			y += i;
		return y;
	}
	
	/**
	 * Subtracts the values of <code>b</code> from <code>a</code>.
	 * 
	 * @param a First tensor.
	 * @param b Second tensor.
	 * @return <code>a -= b</code>.
	 */
	public static double[] difference(double[] a, double[] b) {
		if(a.length != b.length)
			error("Unequal tensor sizes.");
		int x = 0;
		for(double i : b)
			a[x++] -= i;
		return a;
	}
	
	/**
	 * Scales a tensor by a value in-place.
	 * 
	 * @param a Input tensor.
	 * @param b Scalar.
	 * @return <code>a *= b</code>.
	 */
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
		
		final int os  = (is + 2 * pad - fs) / step + 1,
				  isq = is * is,
				  fsq = fs * fs;
		final double[] out = new double[os * os];
		
		for(int or = 0; or < os; or++) {
			final int orst = or * step - pad,
					  oros = or * os;
			for(int oc = 0; oc < os; oc++) {
				final int ocst = oc * step - pad;
				double o = 0.0;
				for(int fr = 0; fr < fs; fr++) {
					final int ir = orst + fr;
					if(ir >= 0 && ir < is) {
						final int iris = ir * is,
								  frfs = fr * fs;
						for(int fc = 0; fc < fs; fc++) {
							final int ic = ocst + fc;
							if(ic >= 0 && ic < is) {
								final int iric = iris + ic,
										  frfc = frfs + fc;
								for(int ch = 0; ch < c; ch++)
									o +=     in[ch * isq + iric]
									   * filter[ch * fsq + frfc];
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
	 * Calculates the gradient of a convolution operation using flattened arrays.
	 * 
	 * @param g Gradient with respect to the output.
	 * @param in Flattened input map.
	 * @param is Input map side length.
	 * @param filter Flattened filter.
	 * @param fs Filter side length.
 	 * @param c Channels.
	 * @param step Step size.
	 * @param pad Padding.
	 * @return The gradient with respect to the input and filter as flattened arrays.
	 */
	public static double[][] backConvolve(double[] g, double[] in, int is, double[] filter, int fs, int c, int step, int pad) {
		final int os  = (is + 2 * pad - fs) / step + 1,
				  isq = is * is,
				  fsq = fs * fs;
		final double[] di = new double[isq * c],
				       df = new double[fsq * c];
		
		for(int or = 0; or < os; or++) {
			final int orst = or * step - pad,
					  oros = or * os;
			for(int oc = 0; oc < os; oc++) {
				final int ocst = oc * step - pad;
				final double l = g[oros + oc];
				for(int fr = 0; fr < fs; fr++) {
					final int ir = orst + fr;
					if(ir >= 0 && ir < is) {
						final int iris = ir * is,
								  frfs = fr * fs;
						for(int fc = 0; fc < fs; fc++) {
							final int ic = ocst + fc;
							if(ic >= 0 && ic < is) {
								final int iric = iris + ic,
										  frfc = frfs + fc;
								for(int ch = 0; ch < c; ch++) {
									final int xi = ch * isq + iric,
											  xf = ch * fsq + frfc;
									di[xi] += l * filter[xf];
									df[xf] += l * in[xi];
								}
							}
						}
					}
				}
			}
		}
		return new double[][] {di,df};
	}
	
	/**
	 * @param x
	 * @return <code>1.0 - x</code> (in-place)
	 */
	public static double[] oneMinus(double[] x) {
		final int xl = x.length;
		for(int i = 0; i < xl; i++)
			x[i] = 1.0 - x[i];
		return x;
	}
	
	/**
	 * Performs an element-wise multiplication in-place.
	 * 
	 * @param a Input tensor.
	 * @param b Scalar tensor.
	 * @return <code>a *= b</code>.
	 */
	public static double[] product(double[] a, double[] b) {
		if(a.length != b.length)
			error("Unequal tensor sizes.");
		int x = 0;
		for(double i : b)
			a[x++] *= i;
		return a;
	}
	
	/**
	 * Performs the dot product of two matrices.
	 * 
	 * @param a An <code>nr x k</code> matrix.
	 * @param b An <code>k x nc</code> matrix.
	 * @param nr Number of rows in matrix <code>a</code>.
	 * @param nc Number of columns in matrix <code>b</code>.
	 * @return An <code>nrxnc</code> matrix where each entry <code>[r][c]</code> is the sum of 
	 * 		   <code>a[r][k] * b[k][c]</code> for all <code>k</code>.
	 */
	public static double[] dot(double[] a, double[] b, int nr, int nc) {
		final int al = a.length,
				  bl = b.length;
		if(al % nr != 0)
			error("Number of rows is not a factor of input \'a\' length.");
		if(bl % nc != 0)
			error("Number of columns is not a factor of input \'b\' length.");
		final int nk = al / nr;
		if(nk != bl / nc)
			error("Number of columns in \'a\' and \'b\' are not equivalent.");
		
		final double[] out = new double[nr * nc];
		for(int r = 0; r < nr; r++) {
			final int oridx = r * nc,
					  aridx = r * nk;
			for(int c = 0; c < nc; c++) {
				double o = 0.0;
				for(int k = 0; k < nk; k++)
					o += a[aridx + k] * b[k * nc + c];
				out[oridx + c] = o;
			}
		}
		return out;
	}
	
	/**
	 * Computes the gradient of the matrix dot product with respect to both
	 * matrices.
	 *  
	 * @param a An <code>nr x k</code> matrix.
	 * @param b An <code>k x nc</code> matrix.
	 * @param l An <code>nr x nc</code> matrix holding the gradient with respect to
	 * 			the output.
	 * @param nr Number of rows in matrix <code>a</code>.
	 * @param nc Number of columns in matrix <code>b</code>.
	 * @return The gradient of <code>a * b</code> with respect to <code>a</code>
	 * 		   and <code>b</code>, respectively.
	 */
	public static double[][] dDot(double[] a, double[] b, double[] l, int nr, int nc) {
		final int al = a.length,
				  bl = b.length;
		if(al % nr != 0)
			error("Number of rows is not a factor of input \'a\' length.");
		if(bl % nc != 0)
			error("Number of columns is not a factor of input \'b\' length.");
		final int nk = al / nr;
		if(nk != bl / nc)
			error("Number of columns in \'a\' and \'b\' are not equivalent.");
		
		return new double[][] {
			dot(
				l,
				transpose2(b,nk,nc),
				nr,
				nk
			),
			dot(
				transpose2(a,nr,nk),
				l,
				nk,
				nc
			)
		};
	}
	
	/**
	 * @param x
	 * @return Sigmoid(x) (in-place)
	 */
	public static double[] sigmoid(double[] x) {
		final int xl = x.length;
		for(int i = 0; i < xl; i++)
			x[i] = 1.0 / (1.0 + Math.exp(-x[i]));
		return x;
	}
	
	/**
	 * @param y Output from sigmoid.
	 * @return Gradient of the sigmoid function (in-place).
	 */
	public static double[] dSigmoid(double[] y) {
		final int yl = y.length;
		for(int i = 0; i < yl; i++)
			y[i] *= 1.0 - y[i];
		return y;
	}
	
	/**
	 * @param x
	 * @return tanH(x) (in-place)
	 */
	public static double[] tanH(double[] x) {
		final int xl = x.length;
		for(int i = 0; i < xl; i++)
			x[i] = Math.tanh(x[i]);
		return x;
	}
	
	/**
	 * @param y Output from tanH.
	 * @return Gradient of the tanH function (in-place).
	 */
	public static double[] dTanH(double[] y) {
		final int yl = y.length;
		for(int i = 0; i < yl; i++)
			y[i] = 1.0 - y[i] * y[i];
		return y;
	}
	
	/**
	 * Prints a formatted array to the console with 1/100 precision.
	 * 
	 * @param in
	 */
	public static void print(double[] in) {
		final NumberFormat fmt = new DecimalFormat("#0.00");
		System.out.print("[ ");
		for(int i = 0; i < in.length; i++)
			System.out.print(fmt.format(in[i]) + " ");
		System.out.print("]");
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

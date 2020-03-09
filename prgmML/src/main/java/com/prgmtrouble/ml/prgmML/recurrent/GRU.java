package com.prgmtrouble.ml.prgmML.recurrent;

import java.io.Serializable;
import java.util.concurrent.ThreadLocalRandom;

import com.prgmtrouble.ml.prgmML.math.Tensor;

/**
 * A lightweight Gated Recurrent Unit object. For the
 * sake of keeping the math simple, I chose not to
 * support customizable functions.
 * 
 * @author prgmTrouble
 */
public class GRU implements Serializable { //TODO use dot product for weights
	/***/
	private static final long serialVersionUID = 1L;
	
	/**Number of weights per gate.*/
	private static final int nWeights = 3;
	/**Weight for the input vector.*/
	private static final int idxW = 0;
	/**Weight for the previous output vector.*/
	private static final int idxU = 1;
	/**Bias vector.*/
	private static final int idxB = 2;
	
	/**Number of gates.*/
	private static final int nGates = 3;
	/**Gate Z.*/
	private static final int idxZ = 0;
	/**Gate H.*/
	private static final int idxH = 1;
	/**Gate R.*/
	private static final int idxR = 2;
	
	/**Number of elements in cache.*/
	private static final int nCacheElements = 7;
	/**Variable cache for each time step.*/
	private double[][][] timeCache;
	/**Weights indexed <code>[gate][weight]</code>.*/
	private double[][][] weights;
	/**Output for <code>time = -1</code>.*/
	private double[] initOut;
	
	/**Number of time steps.*/
	private int maxT;
	/**Size of input vectors.*/
	private int size;
	/**True if the feed-forward operation has been called.*/
	private boolean forwardExecuted = false;
	/**True if the feed-forward operation has been run at least once.*/
	private boolean initialized = false;
	
	/**
	 * Creates a new Gated Recurrent Unit.
	 * <br>
	 * This object does not support customizable functions in order
	 * to keep the math simple.
	 */
	public GRU() {}
	
	/**
	 * Performs the feed-forward operation.
	 * 
	 * @param in Input vectors indexed by time.
	 * @return The outputs indexed by time.
	 */
	public double[][] forward(double[][] in) {
		maxT = in.length;
		size = in[0].length;
		
		if(!initialized) {
			final double v = Math.sqrt(2.0 / (double) (maxT * size));
			final ThreadLocalRandom r = ThreadLocalRandom.current();
			weights = new double[nGates][nWeights][];
			for(int i = 0; i < nGates; i++) {
				final double[][] Wi = new double[nWeights][];
				final boolean isR = i == idxR;
				for(int j = 0; j < nWeights; j++)
					Wi[j] = (isR && j == idxB)? minusOnes(size):gaussian(v,size,r);
				weights[i] = Wi;
			}
			initOut = gaussian(v,size,r);
			//initOut = new double[size];
			initialized = true;
		}
		
		final double[][] out = new double[maxT + 1][]; //Note: indices are shifted up once.
		out[0] = initOut;
		timeCache = new double[maxT][][];
		
		for(int t = 0; t < maxT; t++) {
			final double[][] cache = new double[nCacheElements][];
			final double[] x = cache[0] = in[t],
						  pO = cache[1] = out[t];
			
			double[] az = cache[2] = getAlpha(weights[idxZ][idxW], x, weights[idxZ][idxU], pO, weights[idxZ][idxB]),
					 ar = cache[3] = getAlpha(weights[idxR][idxW], x, weights[idxR][idxU], pO, weights[idxR][idxB]),
					  b = cache[4] = getBeta (weights[idxH][idxW], x, weights[idxH][idxU], pO, ar, weights[idxH][idxB]);
			
			out[t + 1] = 
				Tensor.sum(
					Tensor.product(
						Tensor.dupe(az),
						x
					),
					Tensor.product(
						oneMinus(Tensor.dupe(az)),
						b
					)
				);
			
			timeCache[t] = cache;
		}
		
		forwardExecuted = true;
		
		double[][] fOut = new double[maxT][];
		for(int i = 0; i < maxT; i++)
			fOut[i] = out[i + 1];
		
		return fOut;
	}
	
	/**
	 * Performs the backpropagation operation.
	 * 
	 * @param loss Gradient with respect to the output indexed by time.
	 * @param learningRate Learning rate.
	 * @return Gradient with respect to the input indexed by time.
	 */
	public double[][] backward(double[][] loss, double learningRate) {
		if(!forwardExecuted)
			error("Feed-forward function has not been called for the current cycle.");
		
		double[] dbz = new double[size],
				 dbr = new double[size],
				 dbh = new double[size],
				 dwz = new double[size],
				 dwr = new double[size],
				 dwh = new double[size],
				 duz = new double[size],
				 dur = new double[size],
				 duh = new double[size],
				 dpO = new double[size];
		double[][] dx = new double[maxT][];
		
		for(int t = maxT - 1; t >= 0; t--) {
			double[] l = Tensor.sum(dpO,loss[t]);
			
			double[][] cache = timeCache[t];
			
			double[] c0 = cache[0],
					 c1 = cache[1], //TODO c1 explodes
					 c2 = cache[2],
					 c3 = cache[3],
					 c4 = cache[4];
			
			double[] va = oneMinus(Tensor.dupe(c2));
			double[] vb = 
				Tensor.sum(
					Tensor.scale(Tensor.dupe(c4), -1),
					c1
				);
			double[] v1 =
				Tensor.product(
					Tensor.product(
						Tensor.dupe(c2),
						va
					),
					vb
				);
			double[] v2 =
				Tensor.product(
					oneMinus(
						Tensor.product(
							Tensor.dupe(c4),
							c4
						)
					),
					va
				);
			double[] v3 = 
				Tensor.product(
					oneMinus(Tensor.dupe(c3)),
					c3
				);
			double[] v4 =
				Tensor.product(
					Tensor.dupe(c3),
					weights[idxH][idxU]
				);
			double[] v5 =
				Tensor.product(
					Tensor.product(
						Tensor.dupe(v2),
						v3
					),
					v4
				);
			
			double[] dhdiz = Tensor.product(Tensor.dupe(l), v1);
			//double[][] dwxz = Tensor.dDot(weights[idxZ][idxW], c0, Tensor.dupe(l), nr, nc) //TODO size
			
			
			update(dbz,v1,l);
			update(dbr,v5,l);
			update(dbh,v2,l);
			
			update(dwz,Tensor.product(Tensor.dupe(v1),c0),l);
			update(dwr,Tensor.product(Tensor.dupe(v5),c0),l);
			update(dwh,Tensor.product(Tensor.dupe(v2),c0),l);
			
			update(duz,Tensor.product(Tensor.dupe(v1),c1),l);
			update(dur,Tensor.product(Tensor.dupe(v5),c1),l);
			update(duh,Tensor.product(Tensor.product(Tensor.dupe(v2),c3),c1),l);
			
			dpO =
				Tensor.product(
					Tensor.sum(
						Tensor.sum(
							Tensor.sum(
								Tensor.product(
									Tensor.dupe(vb),
									weights[idxZ][idxU]
								),
								c2
							),
							Tensor.product(
								Tensor.product(
									Tensor.dupe(v2),
									c3
								),
								weights[idxH][idxU]
							)
						),
						Tensor.product(
							Tensor.dupe(v5),
							weights[idxR][idxU]
						)
					),
					l
				);
			
			dx[t] =
				Tensor.product(
					Tensor.sum(
						Tensor.sum(
							Tensor.product(
								Tensor.dupe(v1),
								weights[idxZ][idxU]
							),
							Tensor.product(
								Tensor.dupe(v2),
								weights[idxH][idxW]
							)
						),
						Tensor.product(
							Tensor.dupe(v5),
							weights[idxR][idxW]
						)
					),
					l
				);
		}
		
		update(weights[idxZ][idxB],dbz,learningRate);
		update(weights[idxR][idxB],dbr,learningRate);
		update(weights[idxH][idxB],dbh,learningRate);
		
		update(weights[idxZ][idxW],dwz,learningRate);
		update(weights[idxR][idxW],dwr,learningRate);
		update(weights[idxH][idxW],dwh,learningRate);
		
		update(weights[idxZ][idxU],duz,learningRate);
		update(weights[idxR][idxU],dur,learningRate);
		update(weights[idxH][idxU],duh,learningRate);
		
		update(initOut,dpO,learningRate); //TODO dpO explodes
		
		return dx;
	}
	
	private static void update(double[] a, double[] b, double learningRate) {
		Tensor.sum(
			a,
			Tensor.scale(Tensor.dupe(b), learningRate)
		);
	}
	
	private static void update(double[] a, double[] b, double[] l) {
		Tensor.sum(
			a,
			Tensor.product(
				Tensor.dupe(b),
				l
			)
		);
	}
	
	/**
	 * Creates a new vector with a gaussian distribution.
	 * 
	 * @param v Standard deviation.
	 * @param size Size of vector.
	 * @param r Random object.
	 * @return A vector with elements of a gaussian distribution.
	 */
	private static double[] gaussian(double v, int size, ThreadLocalRandom r) {
		final double[] out = new double[size];
		for(int i = 0; i < size; i++)
			out[i] = r.nextGaussian() * v;
		return out;
	}
	
	private static double[] minusOnes(int size) {
		final double[] out = new double[size];
		for(int i = 0; i < size; i++)
			out[i] = -1.0;
		return out;
	}
	
	/**
	 * Returns the sigmoid output of the weighted input.
	 * 
	 * @param W Input weight vector.
	 * @param xt Input vector.
	 * @param U Previous output weight vector.
	 * @param pO Previous output.
	 * @param b Bias vector.
	 * @return Output of the sigmoid function.
	 */
	private static double[] getAlpha(double[] W, double[] xt, double[] U, double[] pO, double[] b) {
		return
			sigmoid(
				Tensor.sum(
					Tensor.product(
						Tensor.dupe(W),
						xt
					),
					Tensor.sum(
						Tensor.product(
							Tensor.dupe(U),
							pO
						),
						b
					)
				)
			);
	}
	
	/**
	 * Returns the hyperbolic tangent of the weighted input vector and R gate.
	 * 
	 * @param W Input weight vector.
	 * @param xt Input vector.
	 * @param U Previous output weight vector.
	 * @param pO Previous output vector.
	 * @param ar Alpha-R gate output vector.
	 * @param b Bias vector.
	 * @return The output of the hyperbolic tangent function.
	 */
	private static double[] getBeta(double[] W, double[] xt, double[] U, double[] pO, double[] ar, double[] b) {
		return 
			tanH(
				Tensor.sum(
					Tensor.product(
						Tensor.dupe(W),
						xt
					),
					Tensor.sum(
						Tensor.product(
							Tensor.product(
								Tensor.dupe(pO),
								ar
							),
							U
						),
						b
					)
				)
			);
	}
	
	/**
	 * Calculates the output of the GRU cell at the current time.
	 * 
	 * @param az Alpha-Z gate output vector.
	 * @param pO Previous output vector.
	 * @param B Beta gate output vector.
	 * @return The output of the GRU at the current time.
	 */
	private static double[] getOut(double[] az, double[] pO, double[] B) {
		return
			Tensor.sum(
				Tensor.product(
					oneMinus(az),
					pO
				),
				Tensor.product(
					Tensor.dupe(az),
					B
				)
			);
	}
	
	/**
	 * @param in Input vector.
	 * @return <code>in = 1.0 - in</code>
	 */
	private static double[] oneMinus(double[] in) {
		for(int o = 0; o < in.length; o++)
			in[o] = 1.0 - in[o];
		return in;
	}
	
	/**
	 * @param in Input vector.
	 * @return <code>in - 1.0</code>
	 */
	private static double[] minusOne(double[] in) {
		final double[] out = Tensor.dupe(in);
		for(int o = 0; o < out.length; out[o++]--);
		return out;
	}
	
	/**
	 * @param x Input vector.
	 * @return The output of the sigmoid function with input <code>x</code> (in-place).
	 */
	private static double[] sigmoid(double[] x) {
		final int il = x.length;
		for(int i = 0; i < il; i++)
			x[i] = 1.0 / (1.0 + Math.exp(-x[i]));
		return x;
	}
	
	/**
	 * @param x Input vector.
	 * @return The output of the hyperbolic tangent function with input <code>x</code> (in-place).
	 */
	private static double[] tanH(double[] x) {
		final int il = x.length;
		for(int i = 0; i < il; i++)
			x[i] = Math.tanh(x[i]);
		return x;
	}
	
	/**
	 * A custom exception which indicates an error in a GRU.
	 * 
	 * @author prgmTrouble
	 */
	private static class GRUException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "GRU Exception: ";
		
		public GRUException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws a {@linkplain GRUException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	private void error(String s) {
		try {
			throw new GRUException(s);
		} catch (GRUException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}






































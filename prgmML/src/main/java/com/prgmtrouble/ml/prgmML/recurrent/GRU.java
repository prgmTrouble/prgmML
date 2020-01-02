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
public class GRU implements Serializable {
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
	private static final int nCacheElements = 5;
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
				for(int j = 0; j < nWeights; j++)
					Wi[j] = gaussian(v,size,r);
				weights[i] = Wi;
			}
			initOut = gaussian(v,size,r);
			initialized = true;
		}
		
		final double[][] out = new double[maxT + 1][]; //Note: indices are shifted up once.
		out[0] = initOut;
		timeCache = new double[maxT][][];
		
		for(int t = 0; t < maxT; t++) {
			final double[][] cache = new double[nCacheElements][];
			final double[] xt = cache[0] =  in[t],
						   pO = cache[1] = out[t];
			double[][] Wn = weights[idxZ];
			final double[] az = cache[2] = getAlpha(Wn[idxW],xt,Wn[idxU],pO,Wn[idxB]);
			Wn = weights[idxR];
			double[] ar = cache[3] = getAlpha(Wn[idxW],xt,Wn[idxU],pO,Wn[idxB]);
			Wn = weights[idxH];
			final double[] b  = cache[4] = getBeta(Wn[idxW],xt,Wn[idxU],pO,ar,Wn[idxB]);
			ar = null;
			out[t + 1] = getOut(az, pO, b);
			timeCache[t] = cache;
		}
		return out;
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
		double[]   lo = new double[size];
		double[][] lx = new double[maxT][];
		double[] dOdbz = new double[size],
				 dOdbh = new double[size],
				 dOdbr = new double[size],
				 dOdWz = new double[size],
				 dOdWh = new double[size],
				 dOdWr = new double[size],
				 dOdUz = new double[size],
				 dOdUh = new double[size],
				 dOdUr = new double[size];
		for(int t = maxT - 1; t >= 0; t--) {
			double[][] cache = timeCache[t];
			final double[] l = loss[t],
						  xt = cache[0],
						  pO = cache[1],
						  az = cache[2],
						  ar = cache[3],
						  b  = cache[4];
			cache = timeCache[t] = null;
			double[] dOdX = new double[size];
			
			//Gate Z:
			double[][] Wn = weights[idxZ];
			double[] dOdb =
				Tensor.product(
					Tensor.product(
						dOdA(Wn[idxB], pO),
						l
					),
					dAdb(az)
				);
			Tensor.sum(
				dOdbz,
				dOdb
			);
			
			dOdXpO(dOdX, lo, dOdb, Wn[idxW], Wn[idxU]);
			
			dOdUW(dOdUz, dOdWz, xt, pO, dOdb);
			
			//Gate H
			Wn = weights[idxH];
			dOdb =
				Tensor.product(
					Tensor.dupe(l),
					Tensor.product(
						oneMinus(
							Tensor.product(
								Tensor.dupe(b),
								b
							)
						),
						az
					)
				);
			Tensor.sum(
				dOdbh,
				dOdb
			);
			
			dOdXpO(dOdX, lo, dOdb, Wn[idxW], Tensor.product(Tensor.dupe(Wn[idxU]),ar));
			double[] Ut = Tensor.dupe(Wn[idxU]);
			dOdUW(dOdUh, dOdWh, xt, Tensor.product(Tensor.dupe(pO),ar), dOdb);
			
			//Gate R
			Wn = weights[idxR];
			Tensor.product(
				dOdb,
				Tensor.product(
					Ut,
					Tensor.product(
						pO,
						Tensor.product(
							dOdA(Wn[idxB],pO),
							dAdb(ar)
						)
					)
				)
			);
			Ut = null;
			Tensor.sum(
				dOdbr,
				dOdb
			);
			
			dOdXpO(dOdX, lo, dOdb, Wn[idxW], Wn[idxU]);
			Wn = null;
			
			lx[t] = dOdX;
			dOdX = null;
			
			dOdUW(dOdUr, dOdWr, xt, pO, dOdb);
		}
		
		double[][][] dw = new double[][][] {new double[][] {dOdWz,dOdUz,dOdbz},
											new double[][] {dOdWh,dOdUh,dOdbh},
											new double[][] {dOdWr,dOdUr,dOdbr}};
		dOdbz = dOdbh = dOdbr =
		dOdWz = dOdWh = dOdWr =
		dOdUz = dOdUh = dOdUr = null;
		
		for(int g = 0; g < nGates; g++) {
			final double[][] gate = weights[g],
							 grad = dw[g];
			for(int w = 0; w < nWeights; w++)
				update(gate[w],grad[w],learningRate);
		}
		dw = null;
		
		update(initOut,lo,learningRate);
		return lx;
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
								Tensor.dupe(ar),
								U
							),
							pO
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
	 * Calculates the gradient of an Alpha gate with respect to
	 * the bias vector.
	 * <br>
	 * Since the gradient of the weighted input vector with respect
	 * to the bias vector is always <code>1.0</code>, this value
	 * is exactly the same as the gradient of the Alpha gate with
	 * respect to the weighted input vector and therefore may be
	 * used interchangeably.
	 * 
	 * @param a Output of Alpha gate.
	 * @return The gradient of the Alpha gate with respect to the
	 * 		   bias vector.
	 */
	private static double[] dAdb(double[] a) {
		return // dA/dI = A(I)*(A(I)-1)
			Tensor.product(
				minusOne(a),
				a
			);
	}
	
	/**
	 * Calculates the gradient of the output with respect to
	 * the Alpha gate output.
	 * 
	 * @param b Bias vector.
	 * @param pO Previous output vector.
	 * @return Gradient of the output with respect to the Alpha
	 * 		   gate output.
	 */
	private static double[] dOdA(double[] b, double[] pO) {
		return // dO/dA = b - pO
			Tensor.sum(
				Tensor.scale(Tensor.dupe(pO), -1.0),
				b
			);
	}
	
	/**
	 * Calculates the gradient of the output with respect to the
	 * input and previous output vectors in-place.
	 * 
	 * @param dOdX Gradient of the output vector with respect to
	 * 			   the input vector.
	 * @param dOdpO Gradient of the output vector with respect to
	 * 				the previous output vector.
	 * @param dOdb Gradient of the output vector with respect to
	 * 			   the bias vector.
	 * @param W Input weight vector.
	 * @param U Previous output weight vector.
	 */
	private static void dOdXpO(double[] dOdX, double[] dOdpO, double[] dOdb, double[] W, double[] U) {
		Tensor.sum(
			dOdX,
			Tensor.product(
				Tensor.dupe(W),
				dOdb
			)
		);
		
		Tensor.sum(
			dOdpO,
			Tensor.product(
				Tensor.dupe(U),
				dOdb
			)
		);
	}
	
	/**
	 * Calculates the gradient of the output with respect to the
	 * input and previous output weight vectors in-place.
	 * 
	 * @param dU Gradient of the output with respect to the
	 * 			 previous output weight vector.
	 * @param dW Gradient of the output with respect to the
	 * 			 input weight vector.
	 * @param xt Input vector.
	 * @param pO Previous output vector.
	 * @param dOdb Gradient of the output with respect to the
	 * 			   bias vector.
	 */
	private static void dOdUW(double[] dU, double[] dW, double[] xt, double[] pO, double[] dOdb) {
		Tensor.sum(
			dW,
			Tensor.product(
				Tensor.dupe(dOdb),
				xt
			)
		);
		
		Tensor.sum(
			dU,
			Tensor.product(
				Tensor.dupe(dOdb),
				pO
			)
		);
	}
	
	/**
	 * @param in Input vector.
	 * @return <code>1.0 - in</code>
	 */
	private static double[] oneMinus(double[] in) {
		final double[] out = Tensor.dupe(in);
		for(int o = 0; o < out.length; o++)
			out[o] = 1.0 - out[o];
		return out;
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
	 * Updates a vector using SGD in-place.
	 * 
	 * @param in Vector to update.
	 * @param loss Gradient of the output with respect to <code>in</code>.
	 * @param learningRate Learning rate.
	 */
	private static void update(double[] in, double[] loss, double learningRate) {
		Tensor.sum(
			in,
			Tensor.scale(Tensor.dupe(loss), -learningRate)
		);
	}
	
	/**
	 * @param x Input vector.
	 * @return The output of the sigmoid function with input <code>x</code>.
	 */
	private static double[] sigmoid(double[] x) {
		final int il = x.length;
		final double[] out = new double[il];
		for(int i = 0; i < il; i++)
			out[i] = 1.0 / (1.0 + Math.exp(x[i]));
		return out;
	}
	
	/**
	 * @param x Input vector.
	 * @return The output of the hyperbolic tangent function with input <code>x</code>.
	 */
	private static double[] tanH(double[] x) {
		final int il = x.length;
		final double[] out = new double[il];
		for(int i = 0; i < il; i++)
			out[i] = Math.tanh(x[i]);
		return out;
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






































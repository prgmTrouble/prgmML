package com.prgmtrouble.ml.prgmML.recurrent;

import java.io.Serializable;
import java.util.concurrent.ThreadLocalRandom;

import com.prgmtrouble.ml.prgmML.math.Tensor;

public class GatedRecurrentUnit implements Serializable { //TODO update h[0] on a sigmoid.
	
	/***/
	private static final long serialVersionUID = 1L;
	
	private static final int nGates = 3, Z = 0, R = 1, H = 2;
	
	
	private final double[][] W = new double[nGates][];
	
	private final double[][] U = new double[nGates][];
	
	private final double[][] B = new double[nGates][];
	
	private final int[] shape;
	
	private double[][] x;
	
	private double[][] h;
	
	private double[][][] timeCache;
	
	
	
	public GatedRecurrentUnit(int dim1, int dim2, int maxTime) {
		final int inSize = dim1 * dim2;
		final int wtSize = dim1 * dim1;
		
		shape = new int[] {maxTime,dim1,dim2,inSize,wtSize};
		
		final double v = Math.sqrt(2.0 / (double) (wtSize * maxTime));
		ThreadLocalRandom r = ThreadLocalRandom.current();
		
		for(int gate = 0; gate < nGates; gate++) {
			W[gate] = Tensor.gaussian(v, wtSize, r);
			U[gate] = Tensor.gaussian(v, wtSize, r);
			B[gate] = Tensor.newTensor(-1.0, inSize);
		}
		
		h = new double[maxTime + 1][];
		h[0] = Tensor.gaussian(v, inSize, r);
		//h[0] = new double[inSize];
	}
	
	private double[] alpha(double[] W, double[] x, double[] U, double[] h, double[] B) {
		return //Sigmoid(W*x + U*h + b)
			Tensor.sigmoid(
				Tensor.sum(
					Tensor.sum(
						Tensor.dot(
							W,
							x,
							shape[1],
							shape[2]
						),
						Tensor.dot(
							U,
							h,
							shape[1],
							shape[2]
						)
					),
					B
				)
			);
	}
	
	private double[] beta(double[] x, double[] h, double[] r, double[] cache) {
		System.arraycopy(
			Tensor.dot(
				U[H],
				h,
				shape[1],
				shape[2]
			),
			0,
			cache,
			0,
			shape[3]
		);
		return //tanH((U * h) @ r + W * x + B)
			Tensor.tanH(
				Tensor.sum(
					Tensor.sum(
						Tensor.product(
							Tensor.dupe(cache),
							r
						),
						Tensor.dot(
							W[H],
							x,
							shape[1],
							shape[2]
						)
					),
					B[H]
				)
			);
	}
	
	private double[] out(double[] z, double[] h, double[] b, double[] omz) {
		System.arraycopy(
			Tensor.oneMinus(
				Tensor.dupe(z)
			),
			0,
			omz,
			0,
			shape[3]
		);
		return
			Tensor.sum(
				Tensor.product(
					Tensor.dupe(z),
					h
				),
				Tensor.product(
					Tensor.dupe(omz),
					b
				)
			);
	}
	
	public double[][] forward(double[][] in) {
		x = in;
		final int maxT = shape[0],
				  inSize = shape[3];
		timeCache = new double[maxT][][];
		
		for(int t = 0; t < maxT; t++) {
			final double[] xt = x[t],
						   ht = h[t],
						   zt = alpha(W[Z],xt,U[Z],ht,B[Z]),
						   rt = alpha(W[R],xt,U[R],ht,B[R]),
						   ct = new double[inSize],
						  omz = new double[inSize],
						   bt = beta(xt,ht,rt,ct);
			
			h[t + 1] = out(zt, ht, bt, omz);
			timeCache[t] = new double[][] {xt,ht,zt,rt,bt,ct,omz};
		}
		
		return h;
	}
	
	
	
	private void update(double[] a, double[] grad, double lr) {
		Tensor.sum(
			a,
			Tensor.scale(
				Tensor.dupe(grad),
				lr
			)
		);
	}
	
	/**
	 * Runs the backpropagation algorithm.
	 * 
	 * @param gradient Gradient with respect to the outputs, indexed by time.
	 * 				   Note: This value may be mutated.
	 * @param learningRate Learning rate.
	 * @return The gradient with respect to the inputs, indexed by time. 
	 */
	public double[][] backward(double[][] gradient, double learningRate) {
		final int maxT = shape[0],
				  dim1 = shape[1],
				  dim2 = shape[2],
				  inSize = shape[3],
				  wtSize = shape[4];
		final double[][] gradX = new double[maxT][];
		final double[][] gradW = new double[nGates][wtSize];
		final double[][] gradU = new double[nGates][wtSize];
		final double[][] gradB = new double[nGates][inSize];
		
		double[] dh = null;
		
		for(int t = maxT - 1; t >= 0; t--) {
			double[][] cache = timeCache[t];
			
			double[] xt = cache[0],
					 ht = cache[1],
					 zt = cache[2],
					 rt = cache[3],
					 bt = cache[4],
					 ct = cache[5],
					omz = cache[6];
			
			timeCache[t] = cache = null;
			
			double[] dx = new double[inSize];
			
			double[] lt = gradient[t];
			if(t < maxT - 1)
				Tensor.sum(lt,dh);
			
			{
				double[] dhdIz = // ((h - b) @ (z @ (1 - z))) @ d(l)/d(ht)
					Tensor.product(
						Tensor.product(
							Tensor.difference(
								Tensor.dupe(ht),
								bt
							),
							Tensor.product(
								Tensor.dupe(zt),
								omz
							)
						),
						lt
					);
				
				double[][] dWzxt = Tensor.dDot(W[Z], xt, dhdIz, dim1, dim2); // d(Iz)/d({Wz,x})
				
				dx = dWzxt[1];// d(ht)/d(Iz) @ d(Iz)/d(x)
				
				double[][] dUzht = Tensor.dDot(U[Z], ht, dhdIz, dim1, dim2); // d(Iz)/d({Uz,h})
				
				dh = // (d(ht)/d(Iz) @ d(Iz)/d(h)) + z @ d(ht)/d(ht-1)
					Tensor.sum(
						Tensor.product(
							Tensor.dupe(zt),
							lt
						),
						dUzht[1]
					);
				
				Tensor.sum( // d(Wz) += d(Iz)/d(Wz) @ d(ht)/d(Iz)
					Tensor.dupe(gradW[Z]),
					dWzxt[0]
				);
				
				Tensor.sum( // d(Uz) += d(Iz)/d(Uz) @ d(ht)/d(Iz)
					Tensor.dupe(gradU[Z]),
					dUzht[0]
				);
				
				Tensor.sum( // d(Bz) += d(ht)/d(Iz)
					Tensor.dupe(gradB[Z]),
					dhdIz
				);
			}
			
			{
				double[] dhdIb = // ((1 - b @ b) @ (1 - z)) @ d(l)/d(ht)
					Tensor.product(
						Tensor.product(
							Tensor.dTanH(
								Tensor.dupe(bt)
							),
							omz
						),
						lt
					);
				
				double[][] dWbxt = Tensor.dDot(W[H], xt, dhdIb, dim1, dim2); // d(Ib)/d({Wb,x})
				
				Tensor.sum(
					dx,
					dWbxt[1]
				);
				
				double[][] dUbht = Tensor.dDot(U[H], ht, Tensor.product(Tensor.dupe(dhdIb),rt), dim1, dim2); // d(Ib)/d({Ub,h})
				
				Tensor.sum(
					dh,
					dUbht[1]
				);
				
				// d(Wb) += d(Ib)/d(Wb) @ d(ht)/d(Ib)
				Tensor.sum(
					gradW[H],
					dWbxt[0]
				);
				
				// d(Ub) += d(Ib)/d(Ub) @ d(ht)/d(Ib)
				Tensor.sum(
					gradU[H],
					dUbht[0]
				);
				
				// d(Bb) += d(ht)/d(Ib)
				Tensor.sum(
					gradB[H],
					dhdIb
				);
				
				{
					double[] dhdIr = // (r @ (1 - r)) @ (Ub * h)
						Tensor.product(
							Tensor.product(
								Tensor.dSigmoid(
									Tensor.dupe(rt)
								),
								ct
							),
							dhdIb
						);
					
					double[][] dWrxt = Tensor.dDot(W[R], xt, dhdIr, dim1, dim2); // d(Ir)/d({Wr,x})
					
					// dx += ((d(Ib)/d(Ir) @ d(Ir)/d(x)) + (d(Ib)/d(x))') @ d(ht)/d(Ib)
					Tensor.sum(
						dx,
						dWrxt[1]
					);
					
					double[][] dUrht = Tensor.dDot(U[R], ht, dhdIr, dim1, dim2); // d(Ir)/d({Ur,h})
					
					// dh += ((d(Ir)/d(h) @ d(Ib)/d(Ir)) + ((d(Ib)/d(h))' @ r)) @ d(ht)/d(Ib)
					Tensor.sum(
						dh,
						dUrht[1]
					);
					
					// d(Wr) += (d(Ir)/d(Wr) @ d(Ib)/d(Ir)) @ d(ht)/d(Ib)
					Tensor.sum(
						gradW[R],
						dWrxt[0]
					);
					
					// d(Ur) += (d(Ir)/d(Ur) @ d(Ib)/d(Ir)) @ d(ht)/d(Ib)
					Tensor.sum(
						gradU[R],
						dUrht[0]
					);
					
					// d(Br) += d(Ib)/d(Ir) @ d(ht)/d(Ib)
					Tensor.sum(
						gradB[R],
						dhdIr
					);
				}
			}
			gradX[t] = dx;
		}
		
		for(int gate = 0; gate < nGates; gate++) {
			update(W[gate],gradW[gate],learningRate);
			update(U[gate],gradU[gate],learningRate);
			update(B[gate],gradB[gate],learningRate);
		}
		
		update(h[0],dh,learningRate); //TODO gradient w.r.t. h[0].
		
		return gradX;
	}
	
}

















































package com.prgmtrouble.ml.prgmML.light.fcl;

import java.util.concurrent.ThreadLocalRandom;

import com.prgmtrouble.ml.prgmML.light.math.Function;
import com.prgmtrouble.ml.prgmML.light.math.Function.Type;
import com.prgmtrouble.ml.prgmML.math.Tensor;

public class FCLTest {
	
	private static final long print = 10L;
	private static final long max = 10000L;
	private static final int nSets = 5;
	
	private static final int inSize = 5;
	private static final int outSize = 3;
	private static final int nHidden = 2;
	private static final int hiddenSize = 5;
	
	private static final double lr = 0.01;
	
	private static final 	 Type hAct = Type.ReLU;
	private static final Function oAct = new Function(Type.CrossEntropy);
	
	public static void main(String[] args) {
		FCL fcl;
		
		{
			int[] size = new int[nHidden + 2];
			size[0] = inSize;
			size[nHidden+1] = outSize;
			
			for(int i = 1; i < nHidden+1; i++)
				size[i] = hiddenSize;
			
			Function[] activations = new Function[nHidden + 2];
			Function.create(activations, hAct, 0, nHidden+1);
			activations[nHidden+1] = oAct;
			
			fcl = new FCL(activations, size);
		}
		
		ThreadLocalRandom r = ThreadLocalRandom.current();
		
		double[][] sets = new double[nSets][];
		for(int i = 0; i < nSets; i++)
			sets[i] = Tensor.gaussian(1.0, inSize, r);
		int[] exp = new int[nSets];
		for(int i = 0; i < nSets; i++)
			exp[i] = r.nextInt(0, outSize);
		
		double[] in;
		
		long itr = 0;
		boolean converged = false;
		double[] grad = null,out = null;
		while(itr < max && !converged) {
			final int idx = r.nextInt(nSets);
			in = sets[idx];
			
			fcl.forward(Tensor.dupe(in), exp[idx]);
			out = fcl.getOut();
			grad = fcl.backward(lr);
			converged = out[exp[idx]] == 1.00;
			if(itr % print == 0) {
				System.out.print(itr+": out :");
				Tensor.print(out);
				System.out.print(" grad:");
				Tensor.print(grad);
				System.out.print(" converged:"+converged);
				System.out.println();
			}
			itr++;
		}
		
		System.out.println();
		System.out.print(itr+": out :");
		Tensor.print(out);
		System.out.print(" grad:");
		Tensor.print(grad);
		System.out.print(" converged:"+converged);
		System.out.println();
		
		System.out.println();
		System.out.println("in:");
		for(int i = 0; i < nSets; i++) {
			Tensor.print(sets[i]);
			System.out.println("     \t"+exp[i]);
		}
	}
	
	
}















































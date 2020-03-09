package com.prgmtrouble.ml.prgmML.fcl;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.TreeSet;
import java.util.concurrent.ThreadLocalRandom;

/**
 * A <code>HashMap</code> based weight vector object.
 * 
 * @author prgmTrouble
 */
public class WeightVector implements Serializable {
	/***/
	private static final long serialVersionUID = 1L;
	
	/**Weights indexed <code>[destination][source]</code>.*/
	private final HashMap<Integer,HashMap<Integer,Double>> weights;
	
	/**Input received during feed-forward.*/
	private transient double[] in;
	
	/**
	 * Creates a new weight vector.
	 * 
	 * @param initSizeIn Initial size of input.
	 * @param initSizeOut Initial size of output.
	 */
	public WeightVector(int initSizeIn, int initSizeOut) {
		final double a = Math.sqrt(2.0 / (double) (initSizeIn + initSizeOut));
		ThreadLocalRandom r = ThreadLocalRandom.current();
		weights = new HashMap<>(initSizeOut);
		for(int dst = 0; dst < initSizeOut; dst++) { //For each destination:
			final HashMap<Integer,Double> w = new HashMap<>(initSizeIn); //Create destination:
			for(int src = 0; src < initSizeIn; src++) //For each source:
				w.put(src, r.nextGaussian() * a); //Xavier initialization.
			weights.put(dst, w); //Add destination.
		}
	}
	
	/**
	 * Creates a new weight vector.
	 * 
	 * @param initSizeIn Initial size of input.
	 * @param initSizeOut Initial size of output.
	 * @param debug
	 */
	public WeightVector(int initSizeIn, int initSizeOut, boolean debug) {
		weights = new HashMap<>(initSizeOut);
		for(int dst = 0; dst < initSizeOut; dst++) { //For each destination:
			final HashMap<Integer,Double> w = new HashMap<>(initSizeIn); //Create destination:
			for(int src = 0; src < initSizeIn; src++) //For each source:
				w.put(src, (((src + dst) % 2 == 0)? 1.0:-1.0) * (src-dst));
			weights.put(dst, w); //Add destination.
		}
	}
	
	/**@return Size of destination.*/
	private int dstSize() {return weights.size();}
	
	/**
	 * Performs the feed-forward operation.
	 * 
	 * @param in Output of previous layer.
	 * @return Input for next layer.
	 */
	public double[] forward(double[] in) {
		this.in = in;
		final int ds = dstSize(), //Destination size.
				  ss = in.length; //Source size.
		final double[] out = new double[ds]; //Output vector.
		
		for(int dst = 0; dst < ds; dst++) { //For each destination:
			double t = 0.0; //Temporary value.
			final HashMap<Integer,Double> w = weights.get(dst); //Get destination weights.
			for(int src = 0; src < ss; src++) //For each source:
				t += in[src] * w.getOrDefault(src,0.0); //Sum the product of the input and corresponding weight.
			out[dst] = t; //Set output.
		}
		
		return out; //Return output.
	}
	
	/**
	 * Performs the backpropagation procedure with pruning. All weights
	 * whose absolute values are below the threshold will be removed and
	 * all sources and destinations with no attached weights will be recorded
	 * for an implementing layer to handle.<p>
	 * 
	 * Let <code>d</code> be the index for the destination and
	 * <code>s</code> be the index for the source.
	 * @param loss <code>&#120539L[d]/&#120539I'[d]</code>:
	 * 			   The total gradient with respect to the destination' inputs. 
	 * @param learningRate Learning rate.
	 * @param prune Pruning threshold. Set to a negative value to disable.
	 * @return The total gradient with respect to the source's outputs (<code>&#120539L[s]/&#120539O[s]</code>)
	 * 		   as a <code>double[]</code> and a <code>TreeSet</code> containing the indices of any sources (positive)
	 * 		   and destinations (negative) which should be pruned.
	 */
	public Object[] backward(double[] loss, double learningRate, double prune) {
		final int ds = dstSize();
		if(loss == null)
			error("Null Loss.");
		if(loss.length != ds)
			error("Invalid Loss.");
		
		final int ss = in.length; //Source size.
		final boolean p = (prune >= 0.0); //Pruning toggle.
		
		final boolean[] chkSrc = (p)? new boolean[ss] : null; //Should source be removed?
		final double[] di = new double[ss]; // dL/dI[src]
		final TreeSet<Integer> toRemove = (p)? new TreeSet<Integer>() : null; //Set of indices to remove.
		
		for(int dst = 0; dst < ds; dst++) { //For each destination:
			final double l = loss[dst] * learningRate; // dL[dst]/dI'[dst]
			final HashMap<Integer,Double> w = weights.get(dst); // weight[dst]
			
			for(int src = 0; src < ss; src++) { //For each source:
				if(w.containsKey(src)) { //If the destination contains this source:
					if(p && !chkSrc[src]) //If pruning enabled and the source check is false:
						chkSrc[src] = true; //Set this source check to true.
					
					final double t = w.get(src); // weight[dst][src]
					di[src] -= t * l; 					// dL[dst]/dI[src] 		= dL[dst]/dI'[dst] * dI'[dst]/dI[src] 	   = loss[dst] * weight[dst][src]
					final double nw = t - l * in[src]; //  dL[dst]/dW[dst][src] = dL[dst]/dI'[dst] * dI'[dst]/dW[dst][src] = loss[dst] * in[src]
					
					if(!p || Math.abs(nw) > prune) //If pruning disabled or weight is above threshold:
						w.put(src,nw); //Put new value. //TODO (+/-)
					else //Otherwise (pruning is enabled & weight is below threshold):
						w.remove(src); //Remove value.
				}
			}
			if(p && w.size() == 0) //If this destination does not contain a source:
				toRemove.add(-dst); //Add destination.
		}
		if(p)
			for(int src = 0; src < ss; src++) //For each source:
				if(!chkSrc[src]) //If source should be removed:
					toRemove.add(src); //Add source.
		
		return new Object[] {di,toRemove}; //Return gradient and pruning index set.
	}
	
	public Object[] backwardNoLearning(double[] loss, double learningRate, double prune) {
		final int ds = dstSize();
		if(loss == null)
			error("Null Loss.");
		if(loss.length != ds)
			error("Invalid Loss.");
		
		final int ss = in.length; //Source size.
		final boolean p = (prune >= 0.0); //Pruning toggle.
		
		final boolean[] chkSrc = (p)? new boolean[ss] : null; //Should source be removed?
		final double[] di = new double[ss]; // dL/dI[src]
		final TreeSet<Integer> toRemove = (p)? new TreeSet<Integer>() : null; //Set of indices to remove.
		
		for(int dst = 0; dst < ds; dst++) { //For each destination:
			final double l = loss[dst] * learningRate; // dL[dst]/dI'[dst]
			final HashMap<Integer,Double> w = weights.get(dst); // weight[dst]
			
			for(int src = 0; src < ss; src++) { //For each source:
				if(w.containsKey(src)) { //If the destination contains this source:
					if(p && !chkSrc[src]) //If pruning enabled and the source check is false:
						chkSrc[src] = true; //Set this source check to true.
					
					final double t = w.get(src); // weight[dst][src]
					di[src] -= t * l; 					// dL[dst]/dI[src] 		= dL[dst]/dI'[dst] * dI'[dst]/dI[src] 	   = loss[dst] * weight[dst][src]
					final double nw = t - l * in[src]; //  dL[dst]/dW[dst][src] = dL[dst]/dI'[dst] * dI'[dst]/dW[dst][src] = loss[dst] * in[src]
					
					if(!p || Math.abs(nw) > prune) //If pruning disabled or weight is above threshold:
						w.put(src,t); //Put new value. //TODO nw->t
					else //Otherwise (pruning is enabled & weight is below threshold):
						w.remove(src); //Remove value.
				}
			}
			if(p && w.size() == 0) //If this destination does not contain a source:
				toRemove.add(-dst); //Add destination.
		}
		if(p)
			for(int src = 0; src < ss; src++) //For each source:
				if(!chkSrc[src]) //If source should be removed:
					toRemove.add(src); //Add source.
		
		return new Object[] {di,toRemove}; //Return gradient and pruning index set.
	}
	
	/**
	 * Changes the source and/or destinations for weights, usually
	 * after a pruning operation.
	 * 
	 * @param indices An array of indices, indexed
	 * 				  <code>[weight]{old,new}{destination,source}</code>.
	 */
	public void updateStructure(int[][][] indices) {
		final int il = indices.length;
		final double[] t = new double[il];
		for(int i = 0; i < il; i++) {
			final int[] ii = indices[i][0];
			t[i] = weights.get(ii[0]).remove(ii[1]);
		}
		for(int i = 0; i < il; i++) {
			final int[] ii = indices[i][1];
			weights.get(ii[0]).put(ii[1], t[i]);
		}
	}
	
	/**
	 * Deletes the specified destinations from the map (usually
	 * called if a destination index is pruned).
	 * 
	 * @param indices An array of destinations to remove.
	 */
	public void deleteDestination(int[] indices) {
		Arrays.sort(indices); //Sort indices for efficiency.
		final int il = indices.length;
		for(int dst = 0, i = 0; dst < dstSize(); dst++) { //For each destination:
			if(i >= il) //If remove counter exceeds bounds:
				break; //Break.
			if(dst == indices[i]) { //If destination should be pruned:
				weights.remove(dst); //Remove destination.
				i++; //Increment counter.
				continue; //Continue.
			} //Else:
			weights.put(dst - i, weights.remove(dst)); //Move destination up in the list 'i' times (dst & i should be more than zero).
		}
	}
	
	/**
	 * Deletes the specified destinations from the map (usually
	 * called if a destination index is pruned).
	 * 
	 * @param indices A <code>TreeSet</code> of destinations to remove.
	 */
	public void deleteDestination(TreeSet<Integer> indices) {
		final int il = indices.size();
		for(int dst = 0, i = 0; dst < dstSize(); dst++) { //For each destination:
			if(i >= il) //If remove counter exceeds bounds:
				break; //Break.
			if(dst == indices.first()) { //If destination should be pruned:
				weights.remove(dst); //Remove destination.
				i++; //Increment counter.
				indices.pollFirst(); //Remove first element from set.
				continue; //Continue.
			} //Else:
			weights.put(dst - i, weights.remove(dst)); //Move destination up in the list 'i' times (dst & i should be more than zero).
		}
	}
	
	/**
	 * Deletes the specified source from the map (usually
	 * called if a source index is pruned).
	 * 
	 * @param indices An array of sources to remove.
	 */
	public void deleteSource(int[] indices) {
		Arrays.sort(indices); //Sort indices for efficiency.
		final int il = indices.length;
		for(int dst = 0; dst < dstSize(); dst++) { //For each destination:
			final HashMap<Integer, Double> w = weights.get(dst); //Get weights connected to destination.
			for(int src = 0, i = 0; src < w.size(); src++) { //For each source.
				if(i >= il) //If the remove counter exceeds bounds:
					break; //Break.
				if(src == indices[i]) { //If source should be pruned:
					w.remove(src); //Remove source.
					i++; //Increment counter.
					continue; //Continue.
				} //Else:
				w.put(src - i, w.remove(src)); //Move source up in the list 'i' times (src & i should be more than zero).
			}
		}
	}
	
	/**
	 * Deletes the specified source from the map (usually
	 * called if a source index is pruned).
	 * 
	 * @param indices A <code>TreeSet</code> of sources to remove.
	 */
	public void deleteSource(TreeSet<Integer> indices) {
		final int il = indices.size();
		for(int dst = 0; dst < dstSize(); dst++) { //For each destination:
			final HashMap<Integer, Double> w = weights.get(dst); //Get weights connected to destination.
			for(int src = 0, i = 0; src < w.size(); src++) { //For each source:
				if(i >= il) //If the remove counter exceeds bounds:
					break; //Break;
				if(src == indices.first()) { //If source should be pruned: 
					w.remove(src); //Remove source.
					i++; //Increment counter.
					indices.pollFirst(); //Remove element from set.
					continue; //Continue.
				} //Else:
				w.put(src - i, w.remove(src)); //Move source up in the list 'i' times (src & i should be more than zero).
			}
		}
	}
	
	/**
	 * A custom exception which indicates an error in the weight vector.
	 * 
	 * @author prgmTrouble
	 */
	private static class WeightVectorException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "WeightVector Exception: ";
		
		public WeightVectorException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws a {@linkplain WeightVectorException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	private void error(String s) {
		try {
			throw new WeightVectorException(s);
		} catch (WeightVectorException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}

















































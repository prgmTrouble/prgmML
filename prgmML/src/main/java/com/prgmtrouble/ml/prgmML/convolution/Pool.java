package com.prgmtrouble.ml.prgmML.convolution;

import java.io.Serializable;

import com.prgmtrouble.ml.prgmML.generic.ListOfTypes;
import com.prgmtrouble.ml.prgmML.generic.Parameter;

/**
 * An object that pools a flattened tensor and distributes
 * the gradient appropriately during the backward pass.
 * 
 * @author prgmTrouble
 */
public class Pool implements Serializable, ConvolutionLayer {
	/***/
	private static final long serialVersionUID = 1L;
	
	/**
	 * Types of pooling operations.
	 * 
	 * @author prgmTrouble
	 */
	public static enum PoolingTypes {
		/**Maximum of section.*/
		Max,
		/**Absolute maximum of section.*/
		MaxAbs,
		/**Average of section.*/
		Avg,
		/**Maximum of channel.*/
		MaxC,
		/**Absolute maximum of channel.*/
		MaxAbsC,
		/**Average of channel*/
		AvgC;
	}
	
	/**Side length of input map.*/
	private final int s;
	/**Side length of input map, squared.*/
	private final int sq;
	/**Side length of pooled map.*/
	private final int p;
	/**Side length of pooled map, squared.*/
	private final int pq;
	/**Number of channels.*/
	private final int c;
	/**Pooling constant.*/
	private final int f;
	/**Pooling constant, squared.*/
	private final int fq;
	/**Size of flattened input map.*/
	private final int is;
	/**Size of flattened output.*/
	private final int os;
	/**Pooled indices.*/
	private final int[] i;
	/**Pooled tensor.*/
	private final double[] o;
	/**Type of pooling operation.*/
	private final PoolingTypes t;
	/**True if the function pools across channels.*/
	private final boolean channelPooling;
	/**True if the function pools using averages.*/
	private final boolean avgType;
	/**Type for creating new {@linkplain Parameter} objects.*/
	private static final ListOfTypes outputType = new ListOfTypes(new Class<?>[] {double[].class});
	
	/**
	 * Creates a new pooling layer.
	 * 
	 * @param size Size of input map.
	 * @param channels Number of channels.
	 * @param factor Pooling constant.
	 * @param type Pooling operation type.
	 */
	public Pool(int size, int channels, int factor, PoolingTypes type) {
		s = size;
		sq = s * s;
		f = factor;
		c = channels;
		switch(t = type) {
		case MaxC	:
		case MaxAbsC:
		case AvgC	: channelPooling = true;
					  if(c % f != 0)
						  error("Pooling constant is not a factor of the input channels.");
					  p = c / f;
					  os = p * sq;
					  pq = fq = 0;
					  break;
		default		: channelPooling = false;
					  if(s % f != 0)
						  error("Pooling constant is not a factor of the input size.");
					  p = s / f;
					  pq = p * p;
					  fq = f * f;
					  os = c * pq;
					  break;
		}
		is = c * sq;
		o = new double[os];
		i = (avgType = (t == PoolingTypes.Avg || t == PoolingTypes.AvgC))? null:new int[os];
	}
	
	/**
	 * Performs the pooling operation on a flattened input map.
	 * 
	 * @param in Flattened input map.
	 * @return Flattened output map.
	 */
	public double[] forward(double[] in) {
		if(!channelPooling) { //If not pooling channels:
			for(int ch = 0; ch < c; ch++) { //For each channel:
				final int chsq = ch * sq, //Input channel index.
						  chpq = ch * pq; //Pooled channel index.
				for(int pr = 0; pr < p; pr++) { //For each pooled row:
					final int prf = pr * f, //Pool factored row index.
							  prp = chpq + (pr * p); //Pooled row index.
					for(int pc = 0; pc < p; pc++) { //For each pooled column:
						final int pcf = pc * f; //Pool factored column index.
						double a = (t == PoolingTypes.Max)? -Double.MAX_VALUE:0.0; //Compare value.
						int x = 0; //Input index.
						for(int ir = 0; ir < f; ir++) { //For each grid row:
							final int xr = chsq + (prf + ir) * s; //Input row index.
							for(int ic = 0; ic < f; ic++) { //For each grid column:
								final int xc = xr + pcf + ic; //Input column index.
								final double b = in[xc]; //Input value.
								switch(t) {
								case Max   : if(b > a) {a = b; x = xc;} 		   break;
								case MaxAbs: if(Math.abs(b) > a) {a = b; x = xc;}  break;
								case Avg   : a += b / (double) fq;				   break;
								default    : error("Invalid pooling type."); 	   break;
								}
							}
						}
						final int oidx = prp + pc; //Output index.
						o[oidx] = a; //Record value.
						if(!avgType)
							i[oidx] = x; //Record index.
					}
				}
			}
		} else { //Pooling channels: TODO chk indices
			for(int ir = 0; ir < s; ir++) { //For each input row:
				final int irs = ir * s; //Input row index.
				for(int ic = 0; ic < s; ic++) { //For each input column:
					final int icx = irs + ic; //Input column index.
					for(int pch = 0; pch < p; pch++) { //For each pooled channel index.
						final int pchf = pch * f; //Pool factored channel index.
						double a = (t == PoolingTypes.MaxC)? -Double.MAX_VALUE:0.0; //Compare value.
						int x = pchf; //Input index.
						for(int ich = 0; ich < f; ich++) { //For each grid channel:
							final int chx = (pchf + ich) * sq + icx; //Channel grid index.
							final double b = in[chx]; //Input value.
							switch(t) {
							case MaxC: if(a < b) {a = b; x = chx;}			   			   break;
							case MaxAbsC: if(Math.abs(a) < Math.abs(b)) {a = b; x = chx;}  break;
							case AvgC: a += b / (double) f; 				  			   break;
							default: error("Invalid pooling type."); 					   break;
							}
						}
						final int oidx = (pch * sq) + icx; //Output index.
						o[oidx] = a; //Record value.
						if(!avgType)
							i[oidx] = x; //Record index.
					}
				}
			}
		}
		return o; //Return pooled map.
	}
	
	/**
	 * Wraps the output of {@linkplain #forward(double[])} in a {@linkplain Parameter}.
	 * 
	 * @param input A {@linkplain Parameter} containing the input tensor (<code>double[]</code>).
	 * @return The pooled output tensor (<code>double[]</code>) followed by any other values in
	 * 		   the input wrapped in a new {@linkplain Parameter}.
	 */
	@Override
	public Parameter<ListOfTypes> forward(Parameter<ListOfTypes> input) {return new Parameter<ListOfTypes>(outputType, new Object[] {forward((double[]) input.getValues()[0])});}
	
	/**
	 * Feeds the gradient to the correct locations on the input map.
	 * 
	 * @param loss Flattened gradient with respect to the pooled output.
	 * @return Flattened gradient with respect to the input.
	 */
	public double[] backward(double[] loss) {
		final double[] out = new double[is];
		switch(t) {
		case Avg:
			{
				for(int ch = 0; ch < c; ch++) { //For each channel:
					final int cpq = ch * pq, //Pooled channel index.
					  	  	  csq = ch * sq; //Input channel index.
					for(int pr = 0; pr < p; pr++) { //For each pooled row:
						final int prc = cpq + (pr * p), //Pooled column index.
								  prf = pr * f; //Pool factored row index.
						for(int pc = 0; pc < p; pc++) { //For each pooled column:
							final int pcf = csq + (pc * f); //Pool factored column index.
							final double g = loss[prc + pc]; //Pooled gradient.
							for(int ir = 0; ir < f; ir++) { //For each grid row:
								final int csr = pcf + (prf + ir) * s; //Input grid row.
								for(int ic = 0; ic < f; ic++) //For each grid column:
									out[csr + ic] = g; //Record gradient.
							}
						}
					}
				}
			} break;
		case AvgC:
			{
				for(int pch = 0; pch < p; pch++) { //For each pooled channel:
					final int pchidx  = pch * sq, //Pooled channel index.
							  pfchidx = pch * f; //Pool factored channel index.
					for(int ich = 0; ich < f; ich++) { //For each grid channel:
						final int ichidx = (pfchidx + ich) * sq; //Input channel index.
						for(int ir = 0; ir < s; ir++) { //For each row:
							final int irs   = ir * s, //Row index.
									  iridx = ichidx + irs, //Input row index. 
									  pridx = pchidx + irs; //Pooled row index.
							for(int ic = 0; ic < s; ic++) //For each column:
								out[iridx + ic] = loss[pridx + ic]; //Expand gradient.
						}
					}
				}
			} break;
		default: for(int x = 0; x < os; x++) out[i[x]] = loss[x]; break; //Record gradient.
		}
		return out; //Return expanded gradient.
	}
	
	/**
	 * Wraps the output of {@linkplain #backward(double[])} in a {@linkplain Parameter}.
	 * 
	 * @param loss A {@linkplain Parameter} containing the gradient tensor with respect
	 * 			   to the output (<code>double[]</code>).
	 * @return The expanded gradient tensor with respect to the input (<code>double[]</code>)
	 * 		   wrapped in a {@linkplain Parameter}.
	 */
	@Override
	public Parameter<ListOfTypes> backward(Parameter<ListOfTypes> loss) {return new Parameter<ListOfTypes>(outputType, new Object[] {backward((double[]) loss.getValues()[0])});}
	
	/**
	 * A custom exception which indicates an error in the pool.
	 * 
	 * @author prgmTrouble
	 */
	private static class PoolingException extends Exception {
		/***/
		private static final long serialVersionUID = 1L;
		private static final String prefix = "Pooling Exception: ";
		
		public PoolingException(String s) {super(prefix + s);}
	}
	
	/**
	 * Throws a {@linkplain PoolingException} and terminates execution.
	 * 
	 * @param s Description of error.
	 */
	protected static void error(String s) {
		try {
			throw new PoolingException(s);
		} catch(PoolingException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}























































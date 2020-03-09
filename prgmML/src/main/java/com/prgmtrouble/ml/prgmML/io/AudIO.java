package com.prgmtrouble.ml.prgmML.io;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.stream.Stream;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * A class which manages a directory containing <code>.wav</code> audio files.
 * It can read and write audio files as well as make clips and format them into
 * a matrix suitable as inputs to a machine learning algorithm.
 * 
 * @author prgmTrouble
 */
public class AudIO {
	/**Main file directory.*/
	public static String dir;
	/**Length of clip in seconds.*/
	public static int duration;
	/**Number of original files.*/
	private static int nO;
	/**Number of clips.*/
	private static int nC;
	/**Number of arranged clips.*/
	private static int nA;
	/**Original files.*/
	private static File[] original;
	/**Clips.*/
	private static File[] clips;
	/**Arranged clips.*/
	private static File[] arranged;
	
	/**
	 * The id for each file array.
	 * 
	 * @author prgmTrouble
	 */
	public static enum arrayID {original,clips,arranged,all}
	
	/**
	 * Updates the file arrays if changes have been made.
	 * 
	 * @param id The array id.
	 */
	private static void updateFileArray(arrayID id) {
		switch(id) {
		case  original: original = new File(dir + "Original\\").listFiles();return;
		case  	 clips: clips    = new File(dir +    "Clips\\").listFiles();return;
		case  arranged: arranged = new File(dir + "Arranged\\").listFiles();return;
		default		  : original = new File(dir + "Original\\").listFiles();
						clips    = new File(dir +    "Clips\\").listFiles();
						arranged = new File(dir + "Arranged\\").listFiles();return;
		}
	}
	
	/**
	 * Initializes the file directories and settings.
	 * 
	 * @param directory The directory for all saved files.
	 * @param clipLength Length of clip in seconds. This value must be a perfect square to
	 * 					 ensure that the clip can be formatted.
	 * @throws IOException
	 */
	public static void init(String directory, int clipLength) throws IOException {
		dir = directory;
		duration = clipLength;
		
		//Ensure the necessary directories exist.
		new File(dir + "Original\\").mkdirs();
		new File(dir +    "Clips\\").mkdirs();
		new File(dir + "Arranged\\").mkdirs();
		
		//Count the number of files are in each directory.
		try(Stream<Path> files = Files.list(Paths.get(dir+"Original\\"))) {
			nO = (int) files.count(); //Count the number of files in the directory.
			System.out.println("Detected " + nO + " unedited songs."); //Debug.
		}
		try(Stream<Path> files = Files.list(Paths.get(dir+"Clips\\"))) {
			nC = (int) files.count(); //Count the number of files in the directory.
			System.out.println("Detected " + nC + " clips."); //Debug.
		}
		try(Stream<Path> files = Files.list(Paths.get(dir+"Arranged\\"))) {
			nA = (int) files.count(); //Count the number of files in the directory.
			System.out.println("Detected " + nA + " arranged clips."); //Debug.
		}
		//Initialize the file arrays.
		updateFileArray(arrayID.all);
	}
	
	/**
	 * Deletes all the clip data.
	 */
	public static void deleteClips() {
	    File[] contents = new File(dir+"Arranged\\").listFiles();
	    if(contents != null)
	    	for(File f : contents)
	    		if(!Files.isSymbolicLink(f.toPath()))
	    			f.delete();
	    contents = new File(dir+"Clips\\").listFiles();
	    if(contents != null)
	    	for(File f : contents)
	    		if(!Files.isSymbolicLink(f.toPath()))
	    			f.delete();
	    contents = null;
		updateFileArray(arrayID.all);
	}
	
	/**
	 * Writes an object to file.
	 * 
	 * @param i The object.
	 * @param dir The file directory.
	 * @throws IOException
	 */
	private static void write(Object i, String dir) throws IOException {
		try(FileOutputStream   o = new FileOutputStream(dir);
			ObjectOutputStream s = new ObjectOutputStream(o)) {
			s.writeObject(i);
		}
	}
	
	/**
	 * Gets an amplitude as an integer.
	 * 
	 * @param sampleNumber The sample.
	 * @param data The byte array of amplitude data.
	 * @param sampleSize The size of each sample in bytes.
	 * @param channelsNum The number of channels in the audio.
	 * @return The amplitude value.
	 * @throws IOException
	 * @throws UnsupportedAudioFileException
	 */
	private static int getSampleInt(int sampleNumber, byte[] data, int sampleSize, int channelsNum) throws IOException, UnsupportedAudioFileException {
        if (sampleNumber < 0 || sampleNumber >= data.length / sampleSize)
            throw new IllegalArgumentException("sample number can't be < 0 or >= data.length/" + sampleSize + ": " + sampleNumber); //TODO throw a CError
        final byte[] sampleBytes = new byte[4]; //int = 4 bytes.
        for (int i = 0; i < sampleSize; i++) //For each byte in the sample:
            sampleBytes[i] = data[sampleNumber * sampleSize * channelsNum + i]; //Retrieve the data from the byte array
        return ByteBuffer.wrap(sampleBytes).order(ByteOrder.LITTLE_ENDIAN).getInt(); //Transform the sample array into an integer.
    }
	
	/**
	 * Takes a clip from the appropriate directory and formats the data for use with a convolutional network.
	 * 
	 * @param id The clip id.
	 * @throws UnsupportedAudioFileException
	 * @throws IOException
	 */
	private static void formatClip(int id) throws UnsupportedAudioFileException, IOException {
		System.out.println("Arranging clip " + id + "..."); //Debug.
		File f = new File(dir + "Clips\\" + id + ".wav"); //Get a pointer for the input.
		if(!f.exists())
            throw new FileNotFoundException(f.getAbsolutePath()); //TODO throw a CError
		
        AudioInputStream ais = AudioSystem.getAudioInputStream(f); //Create an audio input stream.
        f = null;
        AudioFormat af = ais.getFormat(); //Create a format object for the stream.
        
        final long framesCount = ais.getFrameLength(); //Get the number of frames in the stream.
        final int   sampleSize = af.getSampleSizeInBits() / 8, //Get the size of the sample.
        		   channelsNum = af.getChannels(); //Get the number of channels.
        af = null;
        final long  dataLength = framesCount * sampleSize * channelsNum; //Calculate the length of the .wav file in bytes.
        byte[] data = new byte[(int) dataLength]; //Create the data array.
        
        ais.read(data); //Read the data from the .wav file.
        ais = null;
        
        final int a = (int)(Math.sqrt(data.length / 4) + 2); //Calculate the side length.
		double[] amps = new double[data.length / 4], //Create the amplitude data array.
				fmt[] = new double[a][a]; //Create the formatted data array.
		
		for(int i = 0; i < amps.length; i++) //For each element in the raw data array:
			amps[i] = getSampleInt(i,data,sampleSize,channelsNum); //Get the sample integer.
		data = null;
		
		for(int x = 0; x < a; x++) //For each x coordinate:
			for(int y = 0; y < a; y++) //For each y coordinate:
				fmt[y][x] = (x == 0 || y == 0 || x == a - 1 || y == a - 1)? 0.0:((double) amps[x * (a - 4) + y] / 65535.0); //Set the edges (the buffer region) to zero, and the center area to the data.
		amps = null;
		
		write(fmt,dir + "Arranged\\" + id + ".dji"); //Write the array to the directory.
		fmt = null;
		System.out.println("Clip arranged."); //Debug.
		updateFileArray(arrayID.arranged);
		nA++;
	}
	
	/**
	 * Writes the short value as a 2 byte array in little endian format.
	 * 
	 * @param i The short.
	 * @return The byte array.
	 */
	private static byte[] toByteArr(short i) {
		final byte[] temp = new byte[2]; //Create array to hold the bytes.
		ByteBuffer.wrap(temp).order(ByteOrder.LITTLE_ENDIAN).putShort(i); //Convert input to bytes.
		return temp; //Return the result.
	}
	
	/**
	 * Writes the integer value as a 4 byte array in little endian format.
	 * 
	 * @param i The integer.
	 * @return The byte array.
	 */
	private static byte[] toByteArr(int i) {
		final byte[] temp = new byte[4]; //Create array to hold the bytes.
		ByteBuffer.wrap(temp).order(ByteOrder.LITTLE_ENDIAN).putInt(i); //Convert input to bytes.
		return temp; //Return the result.
	}
	
	/**
	 * Writes a .wav file.
	 * 
	 * @param dir The directory to store the file.
	 * @param data The byte array containing all the amplitude data.
	 * @param sampleRate The sample rate in samples per second (e.g. 44100).
	 * @param sampleSize The number of bits per sample.
	 * @param channels The number of channels (mono = 1, stereo = 2, etc.).
	 * @throws FileNotFoundException
	 */
	public static void write(String dir, byte[] data, float sampleRate, short sampleSize, short channels) throws FileNotFoundException {
		try{
			final DataOutputStream o = new DataOutputStream(new FileOutputStream(dir)); //Create an output stream.
			// write the wav file per the wav file format
            o.writeBytes("RIFF");												// 00 - "RIFF"
            o.write(toByteArr(36 + data.length));								// 04 - Chunk size (total size of file in bytes - 4).
            o.writeBytes("WAVE");												// 08 - "WAVE"
            o.writeBytes("fmt ");												// 12 - "fmt "
            o.write(toByteArr(16));												// 16 - Format chunk size (16 assumes PCM).
            o.write(toByteArr((short) 1));										// 20 - Audio format (1 assumes PCM).
            o.write(toByteArr(channels));										// 22 - Number of channels.
            o.write(toByteArr((int) sampleRate));								// 24 - Samples per second (usually 44100hz).
            o.write(toByteArr((int) (sampleRate * channels * sampleSize / 8)));	// 28 - Bytes per second.
            o.write(toByteArr((short) (channels * sampleSize / 8)));			// 32 - # of bytes in one sample, for all channels.
            o.write(toByteArr(sampleSize));										// 34 - Bits per sample.
            o.writeBytes("data");												// 36 - "data"
            o.write(toByteArr(data.length));									// 40 - Length of the data in bytes.
            o.write(data);														// 44 - The amplitude data.
            o.close(); //Close the data stream.
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Selects a random clip from a .wav file.
	 * 
	 * @param name The file name (excluding extension).
	 * @throws UnsupportedAudioFileException
	 * @throws IOException
	 */
	private static void nextClip(File f) throws UnsupportedAudioFileException, IOException {
		System.out.println("Getting clip...");
		AudioInputStream ais = AudioSystem.getAudioInputStream(f); //Create a stream of data from the file.
		AudioFormat af = ais.getFormat(); //Create a format object for the stream.
		
		//Variables.
		final float    fRate = af.getFrameRate(); if(fRate != 44100) System.out.println("fRate "+fRate+" != 44100."); //Usually 44100hz. TODO Throw an error instead of printing
		final long    frames = ais.getFrameLength(), //The total number of frames in the file.
			           fClip = (long)(fRate * duration); //The length of the clip in frames.
		final int      fSize = af.getFrameSize(), //Size of a frame in bytes.
			         clipLen = (int)(fClip * fSize), //Length of the clip in bytes.
			          bytLen = (int)(frames * fSize), //Length of the original data in bytes.
			          offset = new Random().nextInt((int)((bytLen - clipLen) / fSize)) * fSize; //The random starting position. Dividing and then multiplying by the frame size starts the data on the right byte.
		byte[]    data = new byte[bytLen], //The original data array.
			      clip = new byte[clipLen]; //The clip data array.
		final int time = (int)(offset / fSize / fRate); //Debug.
		String name = f.getName(); //Debug.
		final int x = name.lastIndexOf("."); //Debug.
		if(x > 0) name = name.substring(0,x); //Debug.
		System.out.println("Creating clip " + nC + ".wav: (" +
							String.format("%02d:%02d",time / 60,time % 60) + " - " + 
							String.format("%02d:%02d",(time + 25) / 60,(time + 25) % 60) + 
							") from " + name + "..."); //Debug.
		name = null;
		
		ais.read(data); //Put the amplitude data from the file into the byte array.
		ais = null;
		System.arraycopy(data,offset,clip,0,clipLen); //Copy the portion starting at the random offset into the clip array. TODO use a pointer instead of clip
		
		write(dir + "Clips\\" + nC + ".wav",clip,af.getSampleRate(),(short) af.getSampleSizeInBits(),(short) af.getChannels()); //Write the clip data to a .wav file.
		af = null;
		System.out.println("Clip created."); //Debug.
		updateFileArray(arrayID.clips);
		formatClip(nC++);
	}
	
	/**
	 * Generates a clip from a random song.
	 * 
	 * @throws UnsupportedAudioFileException
	 * @throws IOException
	 */
	public static void nextRandomClip() throws UnsupportedAudioFileException, IOException {nextClip(original[new Random().nextInt(nO)]);}
	
	/**
	 * Prints debug information about a particular .wav file.
	 * 
	 * @param f File to examine.
	 * @throws UnsupportedAudioFileException
	 * @throws IOException
	 */
	public static void info(File f) throws UnsupportedAudioFileException, IOException {
		if(!f.exists())
            throw new FileNotFoundException(f.getAbsolutePath());
		AudioInputStream ais = AudioSystem.getAudioInputStream(f); //Create an audio input stream.
        AudioFormat af = ais.getFormat(); //Create a format object for the stream.
        
        final long framesCount = ais.getFrameLength(); //Get the number of frames in the stream.
        final int   sampleSize = af.getSampleSizeInBits() / 8, //Get the size of the sample.
        		   channelsNum = af.getChannels(); //Get the number of channels.
        af = null;
        final long  dataLength = framesCount * sampleSize * channelsNum; //Calculate the length of the .wav file in bytes.
        byte[] data = new byte[(int) dataLength]; //Create the data array.
        
        ais.read(data); //Read the data from the .wav file.
        ais = null;
        int max = -Integer.MAX_VALUE, min = Integer.MAX_VALUE;
        for(int i = 0; i < data.length / 4; i++) { //For each element in the raw data array:
			int a = getSampleInt(i,data,sampleSize,channelsNum); //Get the sample integer.
			max = Math.max(max, a);
			min = Math.min(min, a);
        }
        data = null;

		String name = f.getName(); //Debug.
		final int x = name.lastIndexOf("."); //Debug.
		if(x > 0) name = name.substring(0,x); //Debug.
		
        System.out.println("Info for " + name + ".wav:  #frames: " + framesCount + 
        										 "  sample size: " + sampleSize + 
        										   "  #channels: " + channelsNum + 
        										      "  #bytes: " + dataLength + 
        									"  #integer samples: " + (dataLength / 4) + 
        									             "  max: " + max + 
        									             "  min: " + min);
	}
	
	/**
	 * Gets a clip from an arranged file.
	 * 
	 * @param n Arranged clip id.
	 * @return The arranged clip as a <code>double[][]</code>.
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public static double[][] readArrangedClip(int n) throws ClassNotFoundException, IOException {
		final File f = new File(dir+"Arranged\\"+n+".dji");
		try(FileInputStream   i = new FileInputStream(f);
			ObjectInputStream o = new ObjectInputStream(i)) {
			return (double[][]) o.readObject();
		}
	}
	
	/**
	 * Gets an array of File objects for the original songs.
	 * 
	 * @return A list of available original songs.
	 */
	public static File[] getOriginals() {return original;}
	/**
	 * Gets an array of File objects for the available clips.
	 * 
	 * @return A list of available clips.
	 */
	public static File[] getClips() {return clips;}
	/**
	 * Gets an array of File objects for the available arranged clips.
	 * 
	 * @return A list of available arranged clips.
	 */
	public static File[] getArranged() {return arranged;}
	/**
	 * Gets the number of original songs.
	 * 
	 * @return The number of original songs.
	 */
	public static int getnO() {return nO;}
	/**
	 * Gets the number of clips.
	 * 
	 * @return The number of clips.
	 */
	public static int getnC() {return nC;}
	/**
	 * Gets the number of arranged clips.
	 * 
	 * @return The number of arranged clips.
	 */
	public static int getnA() {return nA;}
}





















































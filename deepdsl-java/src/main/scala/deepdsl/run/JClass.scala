package deepdsl.run

import deepdsl.cudnn.config._

trait JClass {
	val isFloat = true
}
object JClass extends JClass {
	val (random, gaussian, const, slice) = if (isFloat) ("JTensor.randomFloat", "JTensor.gaussianFloat", "JTensor.constFloat", "JTensor.sliceFloat") 
	                                      else ("JTensor.randomDouble", "JTensor.gaussianDouble", "JTensor.constDouble", "JTensor.sliceDouble") 
	val print = "System.out.println"; val time = "System.nanoTime()"
}
object JTensor extends JClass {
	override def toString = if (isFloat) "JTensorFloat" else "JTensorDouble"
		val get = "get";         val slice = "slice";         val max = "max";     val dot = "dot"
		val flatten = "flatten"; val unflatten = "unflatten"; val exp = "exp";     val log = "log"
		val plus = "plus";       val times = "times";         val tanh = "tanh";   val clip = "clip"
		val sum = "sum";         val call = "call";           val pow = "pow";     val indicator = "indicator"
		val copy = "copy";       val update = "update";
		override val clone = "clone";      
		val asJCudaTensor = "asJCudaTensor";                  val asIndicator = "asIndicator" 
		val accuracy = "accuracy";  val prediction = "prediction"
		// fields
		val dim = "dim";         val array = "array";         val copyOnWrite = "copyOnWrite"
}
object JSlice extends JClass {
	override def toString = "JSlice"
			val addIndexPtn = "addIndexPtn"; val addSlicePtn = "addSlicePtn"; 
			val addShiftPtn = "addShiftPtn"; val addShiftFlipPtn = "addShiftFlipPtn"
					val addUpSamplePtn = "addUpSamplePtn"; val slice = "slice"
}
object JMat extends JClass {
	override def toString = if (isFloat) "JMatFloat" else "JMatDouble"
		val times = "JMatFloatExt.times"; val max = "max"; val sum = "sum"
}
object JMain extends JClass {
	override def toString = if (isFloat) "MainFloat" else "MainDouble"
}
object JCudaMain extends JClass { override def toString = "MainCuda" }
object JType extends JClass {
	val (real, suffix) = if (isFloat) ("float", "f") else ("double", "");  val data = "byte"
}
object JFun extends JClass {
	override def toString = if (isFloat) "FunFloat" else "FunDouble"
		val apply = "apply"
}
object JMath extends JClass {
	val (log, exp, pow, tanh, e) = if (isFloat) 
		("(float) Math.log", "(float) Math.exp", "(float) Math.pow", "(float) Math.tanh", "(float) Math.E")  
		else ("Math.log", "Math.exp", "Math.pow", "Math.tanh", "Math.E") 
}
object JCudaBlas extends JClass { override def toString = "JCudaBlas" } 

object JCudaMatrix extends JClass { 
  override def toString = "JCudaMatrix" 
  val times = "times"; val sum = "sum"; val max = "max"
}

object JCudaTensor extends JClass {
  override def toString = "JCudaTensor"
  val plus_i = "plus_i"; val times_i = "times_i"; val update = "update"; 
  val copy = "copy"; val dot = "dot"; val sum = "sum"; val eq = "eq"
  val asJTensor = "asJTensor"; val asMatrix = "asMatrix"; val flatten = "flatten"; val unflatten = "unflatten"
  override val clone = "clone"; val free = "free";  val log = "log"; val pow = "pow"
  val handle = "handle"; 
  val clearCache = "clearMemoryCache"; val enableCache = "enableMemoryCache"; val disableCache = "disableMemoryCache" 
  val accuracy = "accuracy";  val prediction = "prediction"
}

object JCudaFun extends JClass {
  override def toString = "JCudaFunction"
  val create = "create"
  val destroy = "destroy"
  val free = "free"
  val cacheWorkspace = "enableWorkspaceCache"
  
  val forward = "forward"
  val backward = "backward"; 
  val backward_data = "backward_data"
  val backward_filter = "backward_filter"
  val backward_bias = "backward_bias"
  
  val activation = "JCudnnActivation"
  val pooling = "JCudnnPooling"
  val softmax = "JCudnnSoftmax"
  val convolution = "JCudnnConvolution"
  val lrn = "JCudnnLRN"
  val dropout = "JCudnnDropout"
  val concat = "JCudnnConcat"
  val batchnorm = "JCudnnBatchNorm"
} 

object JActivationMode extends JClass {
  val tanh = "ActivationMode.TANH"
  val sigmoid = "ActivationMode.SIGMOID" 
  val relu = "ActivationMode.RELU" 
  val clipped_relu = "ActivationMode.CLIPPED_RELU"
}
object JSoftmaxAlgorithm extends JClass {
  val accurate = "SoftmaxAlgorithm.ACCURATE"
  val log = "SoftmaxAlgorithm.LOG"
}
object JPoolingType extends JClass {
  val max = "PoolingType.MAX"
  val average = "PoolingType.AVERAGE_EXCLUDE_PADDING"
}
object Stats extends JClass {
	val isStats = "ArithStats.isStats"
	val outStats = "ArithStats.outStats()"
}


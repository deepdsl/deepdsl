package deepdsl.layer

import deepdsl.derivation._
import deepdsl.lang.T.int2Exp
import deepdsl.lang._
import deepdsl.ast._

trait CudaLayer
trait Activation extends CudaLayer
case class Softmax() extends CudaLayer
case class LogSoftmax() extends CudaLayer
case class Convolv(stride: Int, padding: Int) extends CudaLayer
// isMax == true => max pooling, isMax == false => average pooling
case class Pooling(window: Int, stride: Int, padding: Int, isMax: Boolean) extends CudaLayer
case class Tanh() extends Activation
case class ReLU() extends Activation
case class ClippedReLU() extends Activation
case class Sigmoid() extends Activation
case class LRN(n: Int, alpha: Double, beta: Double) extends CudaLayer
case class Dropout(ratio: Float) extends CudaLayer
case class BatchNorm(name: String) extends CudaLayer // batch norm has its own running mean/variance and must be unique

// fix implementation of vec functions 
// xDim: dimensions of input X, yDim: dimensions of output Y, params: parameter variables
object CudaLayer {
    // x: N x K
    def softmax = {
      T.fun(2, (x:VecDec) => FixVec(Softmax(), List(x), x.dim))
    }
    def log_softmax = {
      T.fun(2, (x:VecDec) => FixVec(LogSoftmax(), List(x), x.dim))
    }
    
    def convolv_nobias(name: String, kernelSize: Int, outputChannel: Int, stride: Int, padding: Int) = {
      convolv(name, kernelSize, outputChannel, stride, padding, Param.xavier, Param.fixed(0))
    }
    def convolv(name: String, kernelSize: Int, outputChannel: Int): VecFun = convolv(name, kernelSize, outputChannel, 1, 0)
  
    def convolv(name: String, kernelSize: Int, outputChannel: Int, stride: Int, padding: Int): VecFun = {  
      convolv(name, kernelSize, outputChannel, stride, padding, Param.xavier, Param.const(0))
    }
    
    def convolv(name: String, kernelSize: Int, outputChannel: Int, stride: Int, padding: Int, weight: Param, bias: Param): VecFun = {
    	val C = T.dim; val F = DimConst(outputChannel); val K1 = DimConst(kernelSize); val K2 = DimConst(kernelSize); 
    	val fan_in = C * K1 * K2
      val w = T._new(weight.toInit(fan_in), name + "_W", F, C, K1, K2);  
      val b = T._new(bias.toInit(fan_in), name + "_B", F) 
      convolv(w, b, stride, padding)   
    }
    
    def convolv(w: VecDec, b: VecDec): VecFun = convolv(w, b, 1, 0)
    
    // x: N x C x M1 x M2, w: F x C x K1 x K2, b: F
    def convolv(w: VecDec, b: VecDec, stride: Int, padding: Int) = {
      val N = T.dim; val M1 = T.dim; val M2 = T.dim; 
      val F = w.dim(0); val C = w.dim(1); val K1 = w.dim(2); val K2 = w.dim(3)
      
      T.fun(N, C, M1, M2, x => {
        val s = stride; val p = 2*padding
        val dim = if(stride == 1 && padding == 0) List(N, F, M1-K1+1, M2-K2+1) else List(N, F, (M1-K1+s+p)/s, (M2-K2+s+p)/s)
      
        FixVec(Convolv(stride, padding), List(x, w, b), dim)
      })
    }
    def max_pool(k: Int): VecFun = pooling(k, k, 0, true)
    def max_pool(k: Int, stride: Int, padding: Int) = pooling(k, stride, padding, true)
    def ave_pool(k: Int, stride: Int, padding: Int) = pooling(k, stride, padding, false)
    
    // x: N x C x M1 x M2
    def pooling(k: Int, stride: Int, padding: Int, isMax: Boolean) = { 
      T.fun(4, (x:VecDec) => {
        val s = stride; val p = 2*padding
        val N = x.dim(0); val C = x.dim(1); val M1 = x.dim(2); val M2 = x.dim(3)  
        val dim = if(stride == k && padding == 0) List(N, C, M1/k, M2/k) else List(N, C, (M1-k+s+p)/s, (M2-k+s+p)/s)
        
        FixVec(Pooling(k, stride, padding, isMax), List(x), dim)
      })
    }
    // x: N x C x M1 x M2 or x: N x M
    def tanh(n: Int) = {
      T.fun(n, (x:VecDec) => FixVec(Tanh(), List(x), x.dim))
    }
    def sigmoid(n: Int) = {
      T.fun(n, (x:VecDec) => FixVec(Sigmoid(), List(x), x.dim))
    }
    def relu(n: Int) = {
      T.fun(n, (x:VecDec) => FixVec(ReLU(), List(x), x.dim))
    }
    def clipped_relu(n: Int) = {
      T.fun(n, (x:VecDec) => FixVec(ClippedReLU(), List(x), x.dim))
    }
    def lrn(N: Int, alpha: Double, beta: Double) = {
      T.fun(4, (x:VecDec) => FixVec(LRN(N, alpha, beta), List(x), x.dim))
    }
    // n: number of axis
    def dropout(n: Int, ratio: Float) = {
      T.fun(n, (x:VecDec) => FixVec(Dropout(ratio), List(x), x.dim))
    }
    // gamma is scale and beta is bias
    def batch_norm(name: String, gamma: Float, beta: Float) = { 
      T.fun(4, (x:VecDec) => {
          val C = x.dim(1); val One = T.dim(1)
        	val scale = T._new(ConstInit(gamma), name+"_scale", One, C, One, One)
        	val bias = T._new(ConstInit(beta), name+"_bias", One, C, One, One)
          FixVec(BatchNorm(name), List(x, scale, bias), x.dim)
      })
    }
    
    def concat(lst: Vec*) = ConcatVec(lst.toList)
}


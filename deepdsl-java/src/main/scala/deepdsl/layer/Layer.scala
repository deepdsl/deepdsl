package deepdsl.layer

import deepdsl.derivation._
import deepdsl.lang._
import deepdsl.lang.T.int2Exp
import deepdsl.lang.T.index2Exp
import deepdsl.ast._

object Layer {  
    // y is an indicator matrix
    def loss(y_indicator: VecDec) = {
      val N = y_indicator.dim(0); val K = y_indicator.dim(1) 
      
      T.fun(N, K, (x:VecDec) => 
        (0 - T.sum(N, i => T.sum(K, k => y_indicator(i, k) * x(i, k) ))) / N.size
      )
    }
    
    def log_loss(y_indicator: VecDec) = {
      val N = y_indicator.dim(0); val K = y_indicator.dim(1) 
      
      T.fun(N, K, (x:VecDec) => 
        (0 - T.sum(N, i => T.sum(K, k => y_indicator(i, k) * Log(x(i, k)) ))) / N.size
      )
    }
        
    def precision(y_indicator: VecDec) = {
    	val N = y_indicator.dim(0); val K = T.dim 
      
      T.fun(N, K, (x:VecDec) => 
        T.sum(N, i => 
          Indicator(T.sum(K, k => y_indicator(i, k) * x(i, k)), T.max(K, k => x(i, k)))
        ) / N.size
      )
    } 

    def accuracy(y_label: VecDec, top: Int) = {
      val N = y_label.dim(0); val K = T.dim 
      T.fun(N, K, (x:VecDec) => Accuracy(x, y_label, top)) 
    }
    
    def full(name: String, m2: Int): VecFun = full(name, m2, Param.xavier, Param.const(0))
     
    def full(name: String, m2: Int, weight: Param, bias: Param): VecFun = { 
      val M1 = T.dim; val M2 = DimConst(m2)
      val w = T._new(weight.toInit(M1), name+"_W", M2, M1) 
      val b = T._new(bias.toInit(M1), name+"_B", M2)
      full(w, b)
    }

    // simplified version of full layer
    // W: M2 x M1		B: M2		X: N x M1
    def full(w: VecDec, b: VecDec) = {
      val N = T.dim; val M2 = w.dim(0); val M1 = w.dim(1)
       
      T.fun(N, M1, (x:VecDec) => 
        T.vec(N, M2, (i, j) => 
          T.sum(M1, k => w(j, k) * x(i, k)) + b(j)
        ) 
      )
    }
 	  
	  // flatten dimensions cut, cut+1, .., numOfDim-1, keeps dimensions 0, 1, .., cut-1
    def flatten(numOfDim: Int, cut: Int) = {
      val x = T._new(numOfDim) 
	    val keptIndices= x.getIndices.take(cut)  
	       
      def flat(x: Vec) = {  
        val flatDims = x.getIndices.map(i=>i.dim)
        val j = T.index(flatDims.reduceLeft((c, d) => c * d))
   
  	    VecExp(List(j), x( j.unflatten(flatDims):_* )) 
      }
      val y = flat(x.of(keptIndices:_*))
	    
	    VecFun(x, VecExp(keptIndices ++ y.indices, y.e)) 
	  }
    
    def log_softmax = {
      val N = T.dim; val K = T.dim
      T.fun(N, K, (x:VecDec) => 
        T.vec(N, K, (i, k) => 
          Log( Exp(x(i, k)) / T.sum(K, l => Exp(x(i, l))) )
        )
      )					
    }

    def log_softmax_stable = {
      val N = T.dim; val K = T.dim
      T.fun(N, K, (x:VecDec) => 
        T.vec(N, K, (i, k) => 
          x(i, k) - Log(T.sum(K, l => Exp(x(i, l)))) 
        )
      )	
    }
    
    def loss_with_class_label(y_class_label: VecDec) = {
      val N = y_class_label.dim(0); val K = T.dim 
      
      T.fun(N, K, (x:VecDec) => 
        (0 - T.sum(N, K, (i, k) => 
          Indicator(y_class_label(i), IndexExp(k)) * x(i, k) )
        ) / N.size)
    }
  
    // K: pool size
    def max_pool_NCHW(K: Dim) = {
      val N = T.dim; val F = T.dim; val M1 = T.dim; val M2 = T.dim 
//      val i = T.index("i", N)
//      val i1 = T.index("i1", K); val i2 = T.index("i2", K)
//      val j1 = T.index("j1", M1/K); val j2 = T.index("j2", M2/K);
//      val f = T.index("f", F)
//      
//      val x = T._new("X_p", N, F, M1, M2) 
//      val y = VecExp(List(i, f, j1, j2), Max(Seq(i1, i2), x(i, f, j1 *+ (K.size, i1), j2 *+ (K.size, i2))))  
//      
//      VecFun(x, y)
      T.fun(N, F, M1, M2, x => 
        T.vec(N, F, M1/K, M2/K, (i, f, j1, j2) => 
          T.max(K, K, (i1, i2) => x(i, f, j1 *+ (K.size, i1), j2 *+ (K.size, i2)))    
        )
      )
    }
    
    // K: pool size
    def max_pool_NHWC(K: Dim) = {
      val N = T.dim; val F = T.dim; val M1 = T.dim; val M2 = T.dim 
//      val i = T.index("i", N)
//      val i1 = T.index("i1", K); val i2 = T.index("i2", K)
//      val j1 = T.index("j1", M1/K); val j2 = T.index("j2", M2/K);
//      val f = T.index("f", F)
//      
//      val x = T._new(N, M1, M2, F) 
//      val y = VecExp(List(i, j1, j2, f), Max(Seq(i1, i2), x(i, j1 *+ (K.size, i1), j2 *+ (K.size, i2), f)))  
//      
//      VecFun(x, y)
      T.fun(N, M1, M2, F, x => 
        T.vec(N, M1/K, M2/K, F, (i, j1, j2, f) => 
          T.max(K, K, (i1, i2) => x(i, j1 *+ (K.size, i1), j2 *+ (K.size, i2), f))    
        )
      )
    } 
    
	  // Simplified version of convolv 
	  // W: F x C x K1 x K2		X: N x C x M1 x M2  B: F
	  def convolv_NCHW(w: VecDec, b: VecDec) = {
      val F = w.dim(0); val C = w.dim(1); val K1 = w.dim(2); val K2 = w.dim(3)
      val N = T.dim; val M1 = T.dim; val M2 = T.dim  
      
      T.fun(N, C, M1, M2, x =>  
          T.vec(N, F, M1-K1+1, M2-K2+1, (i, f, j1, j2) => 
            T.sum(C, K1, K2, (c, k1, k2) => x(i, c, j1+k1, j2+k2) * w(f, c, k1, k2)) + b(f)
          )        
      ) 
    }
  
	  // W: K1 x K2 x C x	F 	X: N x M1 x M2 x C    B: F 
	  def convolv_NHWC(w: VecDec, b: VecDec) = {
      val K1 = w.dim(0); val K2 = w.dim(1); val C = w.dim(2); val F = w.dim(3)
      val N = T.dim; val M1 = T.dim; val M2 = T.dim 
//      val x = T._new(N, M1, M2, C)
//      
//      val f = T.index("f", F); val i = T.index("i", N)
//      val j1 = T.index("j1", M1-K1+1); val j2 = T.index("j2", M2-K2+1)
//     
//      val y = T.vec(List(i, j1, j2, f), T.sum("k1", K1, k1 => T.sum("k2", K2, k2 => T.sum("c", C, c => x(i, j1+k1, j2+k2, c) * w(k1, k2, c, f)))) + b(f))
//      
//      VecFun(x, y)
      T.fun(N, M1, M2, C, x =>  
          T.vec(N, M1-K1+1, M2-K2+1, F, (i, j1, j2, f) => 
            T.sum(K1, K2, C, (k1, k2, c) => x(i, j1+k1, j2+k2, c) * w(k1, k2, c, f)) + b(f)
          )        
      ) 
    }
               
//    // W: M2 x M1		B: M2		X: N x M1
//    def full2(w: VecDec, b: VecDec) = {
//      val N = T.dim; val M2 = w.dim(0); val M1 = w.dim(1)
//      val x = T._new("X_f", N, M1); val i = T.index("i", N); val j = T.index("j", M2)
//      
//      val y = VecExp(List(i, j), w.of(j) * x.of(i) + b(j))
//      VecFun(x, y)
//    } 
	  	 	          
//    // W: F x C x K1 x K2		X: N x C x M1 x M2  B: F
//	  def convolv_NCHW_2(w: VecDec, b: VecDec) = {
//      val F = w.dim(0); val C = w.dim(1); val K1 = w.dim(2); val K2 = w.dim(3)
//      val N = T.dim; val M1 = T.dim; val M2 = T.dim 
//      val x = T._new("X_cv", N, C, M1, M2)
//      
//      val f = T.index("f", F); val i = T.index("i", N)    
//      val y = T.vec(List(i, f), T.sumVec("c", C, c => x.of(i, c).convolv(w.of(f, c))) + b(f))
//      
//      VecFun(x, y)
//    }
	     
    // theta: K x M		b: K    Y: N		X: N x M
    def logistic_regression(theta: VecDec, b: VecDec, y: VecDec) = { 
  	  val K = theta.dim(0); val M = theta.dim(1); val N = y.dim(0)
  		val x = T._new(N, M)  
  
  	  val cost = 0 - T.sum(N, i => {
  				  val f_softmax = T.fun(K, (o:VecDec) => 
  				  T.sum(K, 
  						  k => Indicator(y(i), k) * Log( Exp(o(k)) / T.sum(K, l => Exp(o(l))) )
  						  )
  					)					
  					val theta_x = T.vec(K, k => theta.of(k) * x.of(i) + b(k))   
  					f_softmax ( theta_x ) 
  		     }
  		) / N.size
  
  		Vec2ScalarFun(x, cost)
    } 

    def probability(N: Dim, theta: VecDec, b: VecDec) = {
      val K = theta.dim(0); val M = theta.dim(1)    
      val x = T._new(2); val N = x.dim(0)
      
      VecFun(x,  
        T.vec(N, K, (i, k) => {
          val f = T.fun(K, (o:VecDec) => Log( Exp(o(k)) / T.sum(K, l => Exp(o(l)))))    		  			      					
          val theta_x = T.vec(K, k => theta.of(k) * x.of(i) + b(k))  
          f(theta_x)
        })
      )
    }

}

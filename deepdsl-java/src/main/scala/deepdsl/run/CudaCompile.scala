package deepdsl.run

import java.io._
import com.typesafe.config.ConfigFactory
import deepdsl.cudnn.config._
import deepdsl.derivation._
import deepdsl.optimization._ 
import deepdsl.layer._ 
import deepdsl.ast._
import deepdsl.gradient.IR

case class CudaCompile(package_dir: String) {
  var withComments = true; 
	val NANO_FOR_ONE_SEC = 1000000000.0D
  
  val env = new CompileEnv(Map())
	
	private var varCount=0;
  def genVar = { varCount += 1; "y" + varCount }
    
  var constDeclarations: Map[Any, String] = Map() 
  
  // function object -> (fun name, fun declaration)
  // flex_slice object -> (field name, field declaration)
  // constant vector -> (field name, field declaration)
  // (cuda_layer, dims) -> (field name, field declaration)
  var declarations: Map[Any, (String, String)] = Map()
   
	def make_method(rtype: String, name: String, param: String, body: String) = {
	  s"public $rtype $name($param) {\n $body \n}\n"	  
	}
	
	def quoteString(s: String) =  s"""\"$s\""""
  
  def print(loop: Loop) { 
  	  val network_dir = s"src/main/java/$package_dir"
  	  
      var ret = s"package ${package_dir.replace('/', '.')};\n" + 
                "import deepdsl.cudnn.*;\n" +
                "import deepdsl.cudnn.config.*;\n" +
                "import deepdsl.tensor.*;\n" +
                "import deepdsl.util.CudaRun;\n" 
       
      // FIXME: It is important to run this first for the rest to work
      val (train, test) = apply(loop)       
      
      val name = loop.solver.name.capitalize
      
      val (trainData, testData) = loop.data match {
        case Lmdb(dim, train_size, test_size, k) => 
          (s"LmdbFactory.getFactory(${quoteString("dataset/imagenet/ilsvrc12_train_lmdb")}, $train_size, new int[]{${dim.mkString(", ")}}, $k, false)",
           s"LmdbFactory.getFactory(${quoteString("dataset/imagenet/ilsvrc12_val_lmdb")}, $test_size, new int[]{${dim.mkString(", ")}}, $k, true)"   
          ) 
        case Imagenet(dim, train_size, test_size, k) => 
          (s"ImagenetFactory.getFactory(${quoteString("dataset/imagenet/ilsvrc12_train_lmdb")}, $train_size, new int[]{${dim.mkString(", ")}}, $k, false)",
           s"ImagenetFactory.getFactory(${quoteString("dataset/imagenet/ilsvrc12_val_lmdb")}, $test_size, new int[]{${dim.mkString(", ")}}, $k, true)"   
          )                                      
        case Mnist(dim) => 
          (s"MnistFactory.getFactory(true, new int[]{${dim.mkString(", ")}})",
           s"MnistFactory.getFactory(false, new int[]{${dim.mkString(", ")}})"
        )
      }
      
      ret+= s"\n\npublic class $name extends CudaRun {\n\n" +  
             s"public static void main(String[] args){\n"  +                  // start of main
               s"$name run = new $name();\n" +
               s"run.train(${loop.solver.train_itr});\n" +                          // call train
               s"run.test(${loop.solver.test_itr});\n" +                            // call test  
               s"run.save();\n" +
               s"run.free();\n" +
             "}\n\n" +                                                              // end of main
             s"public $name() {\n" +                                           // start of constructor
                s"super(${quoteString(s"$network_dir/${loop.solver.name}")});\n" +
                s"setTrainData($trainData);\n" +
                s"setTestData($testData);\n" +
             s"}\n\n" +                                                        // end of constructor
             s"${printDeclarations}\n\n" +             
             s"$train\n" +        
             s"$test\n" +
            "}"                                        // end of class
       
      val f = new File(s"$network_dir/$name.java") 
      val pw = new PrintWriter(f)
      pw.write(ret)
      pw.close
      
      println(s"file printed to $network_dir/$name.java")
  }
	
	def printDeclarations = constDeclarations.toList
                                             .sortWith((a,b)=>a._1.toString < b._1.toString)
                                             .map({case(key, d) => d})
                                             .mkString("\n") +
                          "\n\n" +
                          declarations.toList.sortWith((a,b)=>a._1.toString < b._1.toString)
                                             .map({case(key, (_,d)) =>  d})
                                             .mkString("\n")
                                             
                                             
  def addConstDeclaration(key: Any, decl: String) = { 
  	      if (!constDeclarations.contains(key)) { 
  	         constDeclarations = constDeclarations + (key -> decl)  
  	      }  
  }
	
	def addDeclaration(key: Any, decl: String => String) = { 
  	      if (!declarations.contains(key)) {
  	         val name = if(key.isInstanceOf[VecParam]) {
  	           val s = key.toString 
  	           if(s.charAt(0).isDigit) {
  	             "_"+s
  	           }
  	           else s
  	         }
  	           else genVar
  	         val x = decl(name)
  	         declarations = declarations + (key -> (name, x)) 
  	         name
  	      } 
  	      else {
  	         declarations(key)._1
  	      }
  }
	
  def apply(loop: Loop): (String, String) = {  
				
    val (t1, t2) = apply(loop.train).runState("") 
		isTraining = false
		val (t3, t4) = apply(loop.test).runState("") 
		
		val (x, y) = loop.input 
		
		(make_method("float", "trainFunction", s"JTensorFloat ${x.toString}, JTensorFloat ${y.toString}", s"$t2\n$t1"), 
		 make_method("JCudaTensor", "testFunction", s"JTensorFloat ${x.toString}", s"$t4\n$t3"))
  } 
	
	private var isTraining = true // we only need this flag for batch normalization where forward call is different for testing
	
  def apply(i: Index): String = env(i)
  def apply(c: Const) : Int = env(c)
  def apply(d: Dim) : Int = env(d).size  
  
  import IR._
  
  def apply(lv: LetVec): MonadState[String, String] = {
	  applyList(lv.lst) >> apply(lv.v) >>= (x => unit(s"return $x;"))
	}
	def apply(lv: LetExp): MonadState[String, String] = {
	  applyList(lv.lst) >> apply(lv.e) >>= (x => unit(s"return $x;"))
	}
  
  def applyList(lst: List[Let]): MonadState[String, Unit] = { 
    val mlst = lst.map({
      case VecLet(x, v) => {  
        apply(v) >>= (tmp1 => unit(s"$JCudaTensor ${x.toString} = $tmp1;"))   
      }
      case MatLet(x, m) => { 
    	  apply(m) >>= (tmp1 => unit(s"$JCudaMatrix ${x.toString} = $tmp1;"))  
      }
      case Update(key, v) => { 
        def f(v: Vec, alpha: AExp, beta: AExp) = apply(key) >>= (k => inplaceUpdate(k, v, alpha, beta) >>= (tmp1 => unit(tmp1 + ";")))
        
        v match {
          case VecPlus(VecTimesScalar(x, b), VecTimesScalar(v, a)) if (key == x) => f(v, a, b) 
          case VecPlus(VecTimesScalar(v, a), VecTimesScalar(x, b)) if (key == x) => f(v, a, b)
          case VecPlus(x, VecTimesScalar(v, a)) if (key == x) => f(v, a, Num(1))
          case VecPlus(VecTimesScalar(v, a), x) if (key == x) => f(v, a, Num(1))
          case VecPlus(VecTimesScalar(x, b), v) if (key == x) => f(v, Num(1), b)
          case VecPlus(v, VecTimesScalar(x, b)) if (key == x) => f(v, Num(1), b)
          case VecPlus(x, v) if (key == x) => f(v, Num(1), Num(1))
          case VecPlus(v, x) if (key == x) => f(v, Num(1), Num(1))
          case _ => f(v, Num(1), Num(0))
        }
      }
      case ExpLet(x, e) => {
        apply(e) >>= (tmp1 => unit(s"${JType.real} ${x.name} = $tmp1;"))
      }
      case Dealloc(v) => { 
        apply(v) >>= (tmp1 => unit(s"$tmp1.${JCudaTensor.free}();" ))
      }

    }) 
    
    (mlst zip lst).foldLeft(unit[String, Unit](()))(
        (c, e) => c >> e._1 >>= (r => modify(s => s"$s${if(withComments) s"// ${e._2}\n" else ""}$r\n"))
    )
    
  }
  
  def inplaceUpdate(param: String, v: Vec, alpha: AExp, beta: AExp): MonadState[String, String] = { 
    apply(alpha) >>= (tmp_a => apply(beta) >>= (tmp_b => (v match {
        case VecProd(m1, m2) => { 
      		apply(m1) >>= (tmp1 => apply(m2) >>= (tmp2 => unit(s"$tmp1.${JCudaMatrix.times}($tmp2, $param, $tmp_a, $tmp_b)")))  
        }
        case SumMat(m) => { 
      	  apply(m) >>= (tmp1 => unit(s"$tmp1.${JCudaMatrix.sum}($param, $tmp_a, $tmp_b)"))
        }
        case FixProd(dy, FixGrad(fv, activeP, gx)) => {
    		  val lst = dy :: activeP  	          
    	        
    	    val mthd =  fv.layer match { 
    	          case Convolv(_,_) if (gx == fv.param(1)) => JCudaFun.backward_filter 
    	          case Convolv(_,_) if(gx == fv.param(2)) => JCudaFun.backward_bias 
    	          case _ => throw new RuntimeException("illegal backward gradient as a part of parameter updates")
    	    } 
    	    apply(lst) >>= (tmp1 => unit(s"${apply(fv)}.$mthd(${tmp1.mkString(", ")}, $param, $tmp_a, $tmp_b)"))
  	    } 
        case VecDec(_, _) => { 
        	apply(v) >>= (tmp1 => unit(s"$param.${JCudaTensor.update}($tmp1, $tmp_a, $tmp_b)"))
        }
    })))
  }
  
  def apply(m: Mat): MonadState[String, String] = {
    m match {
      case m @ MatDec(_,_,_) => unit(m.toString)
      
      // turn into a row-major matrix
      case MatOfVec(idx, vec) => {
        val d = idx.map(i => apply(i.dim))   
  	    apply(vec) >>= (tmp1 => unit(s"$tmp1.${JCudaTensor.asMatrix}(${d.length}, true)"))
      }
      
      /* can't deal with this case */   // case MatExp(idx, vec) => 
      
      case MatOfVecFlexSlice(_, vec) => { 
        if(vec.p.size != 2 || !vec.p(0).isInstanceOf[IndexPattern] || !vec.p(1).isInstanceOf[SlicePattern]) 
          throw new RuntimeException(s"illegal flex slice mat: $m"); 
        apply(vec.v) >>= (tmp1 => unit(s"$tmp1.${JCudaTensor.asMatrix}(1, false)"))
      }
    }
  }
  
  // cache projection variables since they come together as a tuple
  private var projection_cache: Map[Vec, String] = Map()
  // cache batch norm results since they come in triple: d_y, d_scale, and d_bias
  private var batchnorm_cache: Map[Vec, String] = Map()
  
  def apply(lst: List[Vec]): MonadState[String, List[String]] = 
    lst.foldRight[MonadState[String, List[String]]](unit(Nil))((e, c) => c >>= (l => (apply(e) >>= (tmp1 => unit(tmp1 :: l)))) )
  			  
  def apply(v: Vec): MonadState[String, String] = {
	  v match {
	     case x: VecParam => { 
	    	 val p = addDeclaration(x, p => {
	    		 val dim = s"${x.dim.map(d => env(d).size).mkString(", ")}" 

	    		 s"${JCudaTensor} $p = ${if(x.init.fixed) "addFixedParam" else "addParam"}(${quoteString(p)}, ${x.init match {
	    		 case Xavier(_,_) | Caffe_Xavier(_) => s"${quoteString("Random")}, ${env(x.init)}${JType.suffix}, $dim"
	    		 case ZeroInit => s"${quoteString("Constant")}, 0${JType.suffix}, $dim" 
	    		 case ConstInit(x) => s"${quoteString("Constant")}, $x${JType.suffix}, $dim" 
	    		 case Gaussian(std) => s"${quoteString("Gaussian")}, $std${JType.suffix}, $dim"
	    		 }});" 
	    	 })

	    	 unit(p)
	    }

			case x: VecData => unit(x.toString)

  	  case x: VecDecCopy =>  
  	    apply(x.dec) >>= (tmp1 => unit(s"$tmp1.${JCudaTensor.clone}()"))
  	    
  	  case x: VecAsIndicator => {
  	    apply(x.vec) >>= (tmp1 => unit(s"$tmp1.${JTensor.asIndicator}(${apply(x.numOfCls)})"))
  	  }
  	  
  	  case x: CudaVec => { 
  	    apply(x.vec) >>= (tmp1 => unit(s"$tmp1.${JTensor.asJCudaTensor}()"))
  	  }
  	   
  	  case x @ VecDec(_, _) =>  unit(x.toString) 
  	  
  	  case VecClip(v, b) => { 
  	    apply(v) >>= (tmp1 => apply(b) >>= (tmp2 => unit(s"$tmp1.${JTensor.clip}($tmp2)")))
  	  }
  	  
  	  case x @ ConcatVec(lst: List[Vec]) => { 
  	    apply(lst) >>= (tmp1 => unit(s"${apply(x)}.${JCudaFun.forward}(${tmp1.mkString(",")})"))
  	  } 
  	  
  	  case ProjVec(dy: Vec, lst: List[Vec], n: Int) => { 
  	    val (tuple, projection) = (
    	    if(! projection_cache.contains(dy)) {
    	      val proj_tmp = genVar  
    	      projection_cache = projection_cache + (dy -> proj_tmp)
    	      
    	      val dy_tmp = genVar
    	      val proj = 
    	        apply(dy) >>= (tmp1 => unit(s"$JCudaTensor[] $proj_tmp = ${apply(ConcatVec(lst))}.${JCudaFun.backward}($tmp1);\n"))
    	      
    	      (proj_tmp, proj)
    	    }
    	    else
    	      (projection_cache(dy), unit[String, String](""))
    	    )
  	    
  	    projection >>= (tmp1 => modify((s:String) => s+tmp1) >> unit(s"$tuple[$n]"))
  	  }
  	  
  	  case FixProd(dy, FixGrad(fv, activeP, gx)) => {
  		  val lst = dy :: activeP     	         
  	    val assigns = apply(lst) 
  	    val f = apply(fv)
  	        
  	    val mthd =  fv.layer match {
  	          case ReLU() | ClippedReLU() | Sigmoid() | Tanh()
  	          | Pooling(_,_,_,_) | Softmax() | LogSoftmax() 
  	          | LRN(_,_,_) | Dropout(_) | BatchNorm(_) => JCudaFun.backward
  	          
  	          case Convolv(_,_) =>  if (gx == fv.param(0)) JCudaFun.backward_data 
  	                                    else if (gx == fv.param(1)) JCudaFun.backward_filter 
  	                                    else JCudaFun.backward_bias 
  	    }
  	    (
  	        if(fv.layer.isInstanceOf[BatchNorm]) {
  	          val (tuple, batchnorm) = {
  	            if(! batchnorm_cache.contains(dy)) {
  	              val tmp = genVar
 	              
  	        	    batchnorm_cache = batchnorm_cache + (dy -> tmp)  
  	        	    
  	              (tmp, assigns >>= (tmp1 => unit(s"JCudaTensor[] $tmp = $f.$mthd(${tmp1.mkString(",")});\n")))
  	            }
  	            else {
  	              (batchnorm_cache(dy), unit[String, String](""))
  	            }
  	          } 
  	          
  	          batchnorm >>= (tmp1 => (modify((s: String) => s + tmp1) >>
  	            unit(
  	              if(gx == fv.param(0)) {
  	                s"$tuple[0];"
  	              }
  	              else if(gx == fv.param(1)) {
  	                s"$tuple[1];"
  	              }
  	              else {
  	                s"$tuple[2];"
  	              }
  	            )
  	          ))
  	        }
  	        else {  
  	          assigns >>= (tmp1 => unit(s"$f.$mthd(${tmp1.mkString(", ")})"))
  	        }
  	    )
  	  }

  	  case fv @ FixVec(layer, paramlst, _) => {    
  	        val call = JCudaFun.forward + (if(!isTraining && layer.isInstanceOf[BatchNorm]) "_inference" else "") 
  	        apply(paramlst) >>= (tmp1 => unit(s"${apply(fv)}.$call(${tmp1.mkString(", ")})"))
  	  }
  	  
  	  case VecExp(idx, e) => {     
  	      throw new RuntimeException(s"Can't compile VecExp: $v");
  	  }

  	  case FlattenVec(vec, start, d) => { 
  	    apply(vec) >>= (tmp1 => unit(s"$tmp1.${JCudaTensor.flatten}($start, new int[]{${d.map(dim=>apply(dim)).mkString(", ")}})"))
  	  }
  	  case UnFlattenVec(vec, start, d) => { 
  	    apply(vec) >>= (tmp1 => unit(s"$tmp1.${JCudaTensor.unflatten}($start, new int[]{${d.map(dim=>apply(dim)).mkString(", ")}})"))
  	  }
  	  // += copy
  	  case VecPlus(v1, CopyVec(idx, v2)) => {
  		  val d = idx.map(j=>apply(j.dim)).reduce(_*_)  
  	    apply(v2) >>= (tmp2 => apply(v1) >>= (tmp1 => unit(s"$tmp2.${JCudaTensor.copy}($d, $tmp1)"))) 
  	  }
  	  
  	  // += convolution backward_data, += max pooling backward, += average pooling backward
  	  
  	  case VecPlus(v1, FixProd(dy, FixGrad(fv, activeP, gx))) => {
  	    val tmp1 = genVar 

  		  val lst = dy :: activeP       
  	    val mthd =  fv.layer match {
  	          case Pooling(_,_,_,_) => JCudaFun.backward
  	          case Convolv(_,_) if (gx == fv.param(0)) => JCudaFun.backward_data 
  	          case _ => throw new RuntimeException("illegal backward gradient as a part of in-place sum")                         
  	    } 
  	    apply(v1) >>= (tmp1 => apply(lst) >>= (tmp2 => unit(s"${apply(fv)}.$mthd(${tmp2.mkString(",")}, $tmp1)")))
  	  }
  	  
  	  case VecPlus(v1, v2) => { 
  	    apply(v1) >>= (tmp1 => apply(v2) >>= (tmp2 => unit(s"$tmp1.${JCudaTensor.plus_i}($tmp2);")))
  	  }
  	  
  	  case PointwiseProd(v1, v2) => {        
  	    apply(v1) >>= (tmp1 => apply(v2) >>= (tmp2 => unit(s"$tmp1.${JCudaTensor.times_i}($tmp2);")))
  	  }
  	   	  
  	  case VecTimesScalar(vec, e) => {  
  	    apply(vec) >>= (tmp1 => apply(e) >>= (tmp2 => unit(s"$tmp1.${JCudaTensor.times_i}($tmp2);")))
  	  } 

  	  case CopyVec(i, vec) => {
  	    val d = i.map(j=>apply(j.dim)).reduce(_*_) 
  	    apply(vec) >>= (tmp1 => unit(s"$tmp1.${JCudaTensor.copy}($d);")) 
  	  }  
  	  
  	  case VecProd(m1, m2) => {   
  			apply(m1) >>= (tmp1 => apply(m2) >>= (tmp2 => unit(s"$tmp1.${JCudaMatrix.times}($tmp2)")))
  	  } 
  	  
  	  case SumMat(m) => { 
  	    apply(m) >>= (tmp1 => unit(s"$tmp1.${JCudaMatrix.sum}()"))
  	  }
 	  
  	  case LogVec(vec) =>   { 
  	    apply(vec) >>= (tmp1 => unit(s"$tmp1.${JCudaTensor.log}()")) 
  	  }
  	  
  	  case PowVec(vec, x) => { 
  	    apply(vec) >>= (tmp1 => apply(x) >>= (tmp2 => unit(s"$tmp1.${JCudaTensor.pow}($tmp2)")))
  	  } 
  	  
  	  // case ScalarVecApp(f, vec) => 
  	  // case VecApp(f, vec) =>    
      // case VecSlice(vec, idx) =>  
  	  // case VecFlexSlice(vec, p) =>   
  	  // case ExpVec(vec) =>  
  	  case IndicatorVec(v1, v2) =>  { 
  	    apply(v1) >>= (tmp1 => apply(v2) >>= (tmp2 => (unit(s"$tmp1.${JCudaTensor.eq}($tmp2)"))))
  	  }
  	  case MaxMat(m) => { 
  	    apply(m) >>= (tmp1 => unit(s"$tmp1.${JCudaMatrix.max}()"))
  	  }
	  }
  }
  def apply(e: AExp) : MonadState[String, String] = {
	  e match {
	    case x @ ScalarDec(_) => unit(x.toString)  
	    
	    case IndexExp(i) => unit(s"${apply(i)}")
	    
	    case Accuracy(p, y, top) => { 
	      apply(p) >>= (tmp1 => apply(y) >>= (tmp2 => unit(s"$tmp1.${JCudaTensor.accuracy}($tmp2, $top)")))
	    }
	    
	    case r @ Real(_, _) => {
	      addConstDeclaration(r, s"${JType.real} ${r.name} = ${r.x}${JType.suffix};")
	      unit(r.name)
	    }
	    
	    case Indicator(e1, e2) => { 
	      apply(e1) >>= (tmp1 => apply(e2) >>= (tmp2 => unit(s"($tmp1 == $tmp2)? 1 : 0")))
	    } 
	    
	    case ScalarApp(f, a) => {  
	      val fun = if(f == f_tanh) JMath.tanh else apply(f) 
	      
	      apply(a) >>= (tmp1 => unit(s"$fun($tmp1)")) 
	    } 
	    
//	    case Vec2ScalarApp(f, vec) =>  
	    
	    case E => unit(JMath.e)				// evaluate this first since it is a constant
	    
	    case c: Const => unit(apply(c) + "f")
	    
	    case Exp(e1) => { 
	      apply(e1) >>= (tmp1 => unit(s"${JMath.exp}($tmp1)"))
	    }
	    
	    case Log(e1) => { 
	      apply(e1) >>= (tmp1 => unit(s"${JMath.log}($tmp1)"))
	    }
	    
	    case Plus(Num(0), Times(e2, Num(-1))) => { 
	      apply(e2) >>= (tmp1 => unit(s"- $tmp1")) 
	    }
	    
	    case Plus(e1, Times(e2, Num(-1))) => {
	      apply(e1) >>= (tmp1 => apply(e2) >>= (tmp2 => unit(s"$tmp1 - $tmp2"))) 
	    }
	    
	    case Plus(e1, e2) => {  
	      apply(e1) >>= (tmp1 => apply(e2) >>= (tmp2 => unit(s"$tmp1 + $tmp2")))
	    }
	    
	    case Times(e1, Pow(e2, Num(-1))) => {  
	      apply(e1) >>= (tmp1 => apply(e2) >>= (tmp2 => unit(s"$tmp1 / $tmp2")))
	    }
	    
	    case Times(e1, e2) => {  
	      apply(e1) >>= (tmp1 => apply(e2) >>= (tmp2 => unit(s"$tmp1 * $tmp2")))
	    }
	    
	    case Pow(e1, Num(-1)) => {  
	      apply(e1) >>= (tmp1 => unit(s"1 / $tmp1")) 
	    }
	    
	    case Pow(e1, x) => {  
	      apply(e1) >>= (tmp1 => apply(x) >>= (tmp2 => unit(s"${JMath.pow}($tmp1, $tmp2)")))
	    }
	    
	    case Delta(i, j, e1) => {
	      apply(e1) >>= (tmp1 => unit(s"(${apply(i)} == ${apply(j)}) ? $tmp1 : 0"))
	    } 
	    
	    case VecElem(v, indices) => { 
	      apply(v) >>= (tmp1 => unit(s"$tmp1.${JCudaTensor.asJTensor}.${JTensor.get}(${(indices.map(i=>apply(i))).mkString(", ")})"))
	    } 
	    
	    case Max(idx, e) => {
	      val n = idx.map(i=> i.toString) 
	      val d = idx.map(i => apply(i.dim))
	      val max = genVar
	    	val tmp = genVar
	    	
	    	val x = apply(e) >>= (tmp1 => unit(
  	      s"${JType.real} $tmp;\n" +
  	      s"${JType.real} $max = 0;\n" + 
  	      (n zip d).foldRight(s"$tmp = $tmp1\nif($max < $tmp) { $max = $tmp; }")(
  	          (e, c) => s"for(int ${e._1} = 0; ${e._1} < ${e._2}; ${e._1} ++) {\n$c\n}") + "\n"
	      ))
	      
	      x >>= (tmp1 => modify((s:String) => s + tmp1) >> unit(s"$max"))
	    } 
	    
	    case Sum(i, e) => {
	    	val tmp = genVar 
	      val n = i.toString 
	      val sum = genVar
	      
	      val x = apply(e) >>= (tmp1 => unit(
  	      s"${JType.real} $tmp;\n" + 
  	      s"${JType.real} $sum = 0;\n" + 
  	      s"for(int $n = 0; $n < ${apply(i.dim)}; $n ++) {\n" +
     	      s"$tmp = ${}\n" + 
    	      s"$sum += $tmp;\n" +
  	      s"}\n"
	      ))
	      x >>= (tmp1 => modify((s: String) => s + tmp1) >> unit(s"$sum"))
	    } 
	    
	    case InnerProd(v1, v2) => {
	      apply(v1) >>= (tmp1 => apply(v2) >>= (tmp2 => unit(s"$tmp1.${JCudaTensor.dot}($tmp2)")))
	    } 
//	    case MaxVec(v) =>  
	    
	    case SumVec(v) => { 
	      apply(v) >>= (tmp1 => unit(s"$tmp1.${JCudaTensor.sum}()"))
	    }
	  }
	}
  
  def apply(f: ScalarFun): String = addDeclaration(f, fn => { 
	  val (s, tmp1) = apply(f.body).runState("") 

		s"${JType.real} $fn(${JType.real} ${f.param.toString}) {\n $s \n return $tmp1;\n}" 

  })
  
  def apply(fv: ConcatVec): String = {
    val dimlst = fv.lst.map(x => x.getDims.map(d => apply(d)))
    val dimParam = dimlst.map (d => s"new int[]{${d.mkString(",")}}"). mkString(",")
  	addDeclaration(("concat", dimlst), f => s"${JCudaFun.concat} $f = addConcat($dimParam);" )
  }
  
  def apply(fv : FixVec): String = { 
	  val dimlst = fv.param.map(x => x.getDims.map(d => apply(d)))  
		val dimParam = dimlst.map (d => s"new int[]{${d.mkString(",")}}"). mkString(",")
	  val key = if(fv.layer.isInstanceOf[Dropout]) fv else (fv.layer, dimlst)
    addDeclaration(key, f => {
              val l = quoteString(f)
    	        fv.layer match {
      	        case Tanh() => s"${JCudaFun.activation} $f = addActivation($dimParam, ${JActivationMode.tanh});" 
      	        case Sigmoid() => s"${JCudaFun.activation} $f = addActivation($dimParam, ${JActivationMode.sigmoid});" 
      	        case ReLU() => s"${JCudaFun.activation} $f = addActivation($dimParam, ${JActivationMode.relu});" 
      	        case ClippedReLU() => s"${JCudaFun.activation} $f = addActivation($dimParam, ${JActivationMode.clipped_relu});" 
      	        case Pooling(w,s,p,isMax) => s"${JCudaFun.pooling} $f = addPooling($dimParam, $w, $s, $p, ${if(isMax) JPoolingType.max else JPoolingType.average});"   
      	        case Softmax() => s"${JCudaFun.softmax} $f = addSoftmax($dimParam, ${JSoftmaxAlgorithm.accurate});" 
      	        case LogSoftmax() => s"${JCudaFun.softmax} $f = addSoftmax($dimParam, ${JSoftmaxAlgorithm.log});" 
      	        case Convolv(s,p) => s"${JCudaFun.convolution} $f = addConvolution($dimParam, $s, $p);" 
      	        case LRN(n, alpha, beta) => s"${JCudaFun.lrn} $f = addLRN($dimParam, $n, $alpha, $beta);"
      	        case Dropout(ratio) => s"${JCudaFun.dropout} $f = addDropout($l, $dimParam, $ratio${JType.suffix});"
      	        case BatchNorm(name) => s"${JCudaFun.batchnorm} $f = addBatchNorm($l, new int[]{${dimlst(0).mkString(",")}});"
    	        }
    }) 
  }
}
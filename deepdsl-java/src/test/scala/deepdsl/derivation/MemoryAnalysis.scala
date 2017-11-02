package deepdsl.derivation

import deepdsl.optimization._
import deepdsl.layer._
import deepdsl.run._
import deepdsl.analysis._
import deepdsl.ast._

object MemoryAnalysis {
  private val env = new Env(Map())
  private val mb = 1E6 
  
  private def workspaceMemory(lst: List[Let]) = {
    val convolutions = 
      lst.foldLeft[Set[FixVec]](Set())((c, let) => let match {
        case VecLet(x, v @ FixVec(Convolv(_,_),_,_)) => c + v
        case _ => c
      }).toList.sortWith((a,b) => a.param(1).toString < b.param(1).toString)  
     
    def getWorkspace(cv: FixVec) = {
      val c = cv.layer.asInstanceOf[Convolv]
      val stride = c.stride; val padding = c.padding;
      val x_dim = cv.param(0).getDims.map(d=>env(d).size).toArray
      val w_dim = cv.param(1).getDims.map(d=>env(d).size).toArray
      val b_dim = cv.param(2).getDims.map(d=>env(d).size).toArray
      val jcv = new deepdsl.cudnn.JCudnnConvolution(x_dim, w_dim, b_dim, stride, padding)
      val size = jcv.workspaceSize().map(s => s/mb)
      (cv, size)
    }
    val workspaceMap = convolutions .map(cv => getWorkspace(cv))
    val max = deepdsl.cudnn.JCudaFunction.getWorkspaceSize()/mb
    
//    printf("\n%-20s%-20s%-20s%-20s\n", "workspace", "forward", "backward data", "backward filter")
//    val workspace =  workspaceMap.map({case (cv, size) => 
//      f"${cv.param(1).asInstanceOf[VecDec].v.name.split("_")(0)}%-20s${size(0)}%-20s${size(1)}%-20s${size(2)}%-20s"}).mkString("\n")
//    println(workspace)
//    println(s"\nconvolution workspace: $max")
    (workspaceMap.toMap, max)
  }
   
  def parameterMemory(param: List[VecParam], momentum: Float) {  
    val size = param.map(e => e.dim.map(d => env(d).size).reduce(_*_) * 4 / mb)
    
//    println
//    for((v, s) <- param zip size) {
//      printf("%-60s%20s\n", v, s)
//    }
    
    val total = size.reduce(_+_)
    
    println(s"\ntotal parameter memory: ${total}")
    if(momentum > 0) {
      println(s"with SGD, this doubles to ${total * 2}\n")
    }
  }
  
  def runtimeMemory(lst: List[Let]) = {
    val sizes = Memory(lst)._1.map(x => x/mb)
    var pool: Set[Double] = Set()
    
    val total_sizes = sizes.foldLeft[(List[(Double, Double)], Double, Double)]((Nil, 0, 0))((c, e) => {
      val x = c._2 + e; 
      if (e < 0) {
        pool = pool + (-e)
      }
      val y = if(e > 0) {
        if(pool.contains(e)) {
          pool = pool - e
          c._3
        }
        else {
          val l = pool.filter( p => e <= p && p * 0.7 <= e ).toList
          if(l.size > 0) {
            pool = pool - l(0)
            c._3
          }
          else 
            c._3 + e 
        }
      }  
      else 
        c._3; 
      
      (c._1:::List((x, y)), x, y)}
      
    )._1
    
        
    def getWorkspace(map: Map[FixVec, Array[Double]], v: Vec) = {
      v match {
        case f@FixVec(Convolv(_,_), _,_) => map(f)(0)
        case FixProd(_, FixGrad(f@FixVec(Convolv(_,_), param,_), _,x)) => {
          if (x == param(0)) map(f)(1)
          else if (x == param(1)) map(f)(2)
          else 0
        } 
        case _ => 0
      }
    }
  
    val (workspaceMap, workspaceMax) = workspaceMemory(lst)
    
    val workspaceSizes = lst.map(l => l match {
                    case VecLet(_,v) => getWorkspace(workspaceMap, v)
                    case Update(_,VecPlus(VecTimesScalar(v, _), _)) => getWorkspace(workspaceMap, v)
                    case _ => 0})
     
    printf("\n%-85s%-15s%20s%20s%20s%20s\n\n", "IR Expression", "Dimensions", "Workspace", "Current mem.", "Total dyn. mem.", "Total cached mem.")

    for((l, s) <- lst.zip(workspaceSizes zip sizes zip total_sizes)) {
      printf("%-85s%-15s%20s%20f%20f%20f\n", l, 
          (l match {case VecLet(x,_) => x.dim.map(d=>env(d).size).mkString(" ") 
                    case _ => ""}), 
           s._1._1,
           s._1._2, s._2._1, s._2._2)
    }
    val (no_cache, cache) = total_sizes.unzip
    val dyn_size = (workspaceSizes zip no_cache) map (x => x._1 + x._2)
    println(s"\nmax dynamic memory: ${cache.last + workspaceMax}")
    println(s"\nmax dynamic memory without tensor cache: ${no_cache.max + workspaceMax}")
    println(s"\nmax dynamic memory without tensor or workspace cache: ${dyn_size.max}")
  }
}
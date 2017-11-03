package deepdsl.optimization

import deepdsl.derivation._
import deepdsl.run._
import deepdsl.analysis._
import deepdsl.layer._
import deepdsl.lang._
import deepdsl.ast._
 
case class LetExp(lst: List[Let], e: AExp)
case class LetVec(lst: List[Let], v: Vec)

trait Let {
  def sub(vl: VecDec, vr: Vec) : Let = this match { 
    case ExpLet(x, e) => ExpLet(x, e.sub(vl, vr))
    case Dealloc(_) => this
    case Update(key, x) => Update(key, x.sub(vl, vr))
    case VecLet(x, v) => VecLet(x, v.sub(vl, vr))
    case MatLet(x, m) => MatLet(x, m.sub(vl, vr)) 
  }
  def sub(ml: MatDec, mr: MatDec) : Let = this match {
    case VecLet(x, v) => v match {
      case VecProd(m1, m2) => VecLet(x, VecProd(sub(m1, ml, mr), sub(m2, ml, mr)))
      case SumMat(m) => VecLet(x, SumMat(sub(m, ml, mr)))
      case MaxMat(m) => VecLet(x, MaxMat(sub(m, ml, mr)))
      case _ => this
    } 
    case _ => this
  }
  private def sub(m: Mat, ml: MatDec, mr: MatDec) = if(m == ml) mr else m 
  def contains(y: Vec) : Boolean = this match { 
    case ExpLet(_, e) => e.contains(y)
    case Dealloc(_) => false
    case Update(_, x) => x.contains(y)
    case VecLet(_, v) => v.contains(y)
    case MatLet(_, m) => m.contains(y) 
  }
  def contains(y: MatDec) : Boolean = this match { 
    case ExpLet(_, _) => false
    case Dealloc(_) => false
    case Update(_, x) => x.contains(y)
    case VecLet(_, v) => v.contains(y)
    case MatLet(_, m) => m.contains(y) 
  }
  override def toString = this match { 
    case ExpLet(x, e) => s"val $x = $e"
    case Dealloc(x) => s"dealloc $x"
    case Update(key, x) => s"$key = $x"// s"$key = $beta * $key + $alpha * $x"
    case VecLet(x, v) => s"val $x = $v"
    case MatLet(x, m) => s"val $x = $m"  
  }
}  
case class ExpLet(x: ScalarDec, e: AExp) extends Let  
case class Dealloc(x: VecDec) extends Let  
//// key = beta * key + alpha * x
//case class Update(key: VecParam, x: Vec, alpha: AExp, beta: AExp) extends Let
case class Update(x: VecParam, v: Vec) extends Let
case class VecLet(x: VecDec, v: Vec) extends Let 
case class MatLet(x: MatDec, m: Mat) extends Let  
 
case class SSA(isTraining: Boolean) {  
  def optimize(e: LetExp): LetExp = {
    val (e1, l) = apply1(e.e).runState(Nil)
    LetExp(optimize(e.lst ++ l), e1)
  }
  
  def optimize(v: LetVec): LetVec = {
    val (v1, l) = apply1(v.v).runState(Nil)
    LetVec(optimize(v.lst ++ l), v1)
  }
  
  def optimize (lst: List[Let]) : List[Let] = 
    dealloc(
        markCopy(
            schedule(
                merge(
                    elim(
                        pre_process(lst))))))
     
  def pre_process(lst: List[Let]): List[Let] = removeAlias(lst.flatMap(let => apply(Optimize(let)))) 
  
  def removeAlias(lst: List[Let]) = {
    var i = 0;
    var ret = lst
    while (i < ret.size) {
      val e = ret(i)
      e match {
        case VecLet(x:VecDec, y:CudaVec) =>  
        case VecLet(x:VecDec, y:VecAsIndicator) =>  
        case VecLet(x:VecDec, y:VecData) =>
        case VecLet(x:VecDec, y:VecDec) => ret = ret.map(l => l.sub(x, y)) 
        case VecLet(x:VecDec, y:FlattenVec) => ret = ret.map(l => l.sub(x, y)) 
        case VecLet(x:VecDec, y:UnFlattenVec) => ret = ret.map(l => l.sub(x, y)) 
        case VecLet(x:VecDec, y:VecSlice) => ret = ret.map(l => l.sub(x, y)) 
        case _ => 
      }
      i = i + 1
    } 
    ret.filter({
      case VecLet(x:VecDec, y:CudaVec) => true
      case VecLet(x:VecDec, y:VecAsIndicator) => true
      case VecLet(x:VecDec, y:VecData) => true
      case VecLet(x:VecDec, y:VecDec) => false 
      case VecLet(x:VecDec, y:FlattenVec) => false
      case VecLet(x:VecDec, y:UnFlattenVec) => false 
      case VecLet(x:VecDec, y:VecSlice) => false
      case _ => true
    })
  }

  // if a vec def has one use in update or vec plus, then merge them
  def merge(lst: List[Let]) = {
    val depend = use_dependency(lst)
    
    val single_use_lst = depend.keySet.filter(let => 
    		depend(let) match {
    		  case List(Update(_,v1)) => v1.contains(let.x) && // FIXME: This is unsafe
    		    (let.v match { 
    		      case VecProd(_,_) | SumMat(_) | FixProd(_, FixGrad(FixVec(Convolv(_,_),_,_), _, _)) => true 
    		      case _ => false
    		      })
    		  case List(VecLet(_, VecPlus(_,v1))) => v1 == let.x &&
    		    (let.v match {
    		      case CopyVec(_,_) | FixProd(_, FixGrad(FixVec(Convolv(_,_),_,_), _, _)) 
    		                    | FixProd(_, FixGrad(FixVec(Pooling(_,_,_,_),_,_), _, _)) => true
    		      case _ => false 
    		    })
//    		  case List(VecLet(_, v)) => v == let.x
    		  case _ => false
    		}
    )
    
    lst.filter({ case let@VecLet(_,_) => !single_use_lst.contains(let) case _ => true })
       .map(let => single_use_lst.foldLeft(let)((c, e) => c.sub(e.x, e.v)))
  }
  
  // insert deallocation statement for each VecDec at the earliest point
  // 1. forward pass: collect all VecDec defined in VecDec -> Vec definitions except data loading (VecData)
  // 2. forward pass: remove VecDec reused in-place -- but not copied (VecDecCopy)
  //                  collect all MatDec -> Mat definitions  
  // 3. backward pass: for each statement, find live VecDec used and live MatDec used (and find its VecDec alias among the live set)
  //                                       insert deallocation statement for these VecDec, remove them from live set
  def dealloc(lst: List[Let]) = {
    val lv1 = lst.foldRight(List[VecDec]())((e, c) => e match {
      case VecLet(x, v) if (!v.isInstanceOf[VecData]) => x :: c
      case _ => c
    }) 
    
    val (liveVec, liveMat) = lst.foldLeft((lv1.toSet, Map[MatDec, Mat]()))((c, e) => e match {
      case VecLet(x, v) => { 
        val c1 = v.actIfInplace(v1 => if(!v1.isInstanceOf[VecDecCopy]) c._1.filter(v => !v1.contains(v)) else c._1, c._1)
        (c1, c._2)
      } 
      case MatLet(x, m) => (c._1, c._2 + (x -> m))
      case _ => c
    })
    
    val (ret, _, _) = lst.foldRight((List[Let](), liveVec, liveMat))((e, c) => {
      val (sofar, lv, lm) = c
      val deadVec = lv.filter(v => e.contains(v)) 
      val deadMat = lm.filter(m => e.contains(m._1))    
      val dead = deadVec ++ lv.filter(v => deadMat.exists(m=>m._2.contains(v)))
      (e :: dead.toList.map(d => Dealloc(d)) ++ sofar, lv -- dead, lm -- deadMat.keySet)
    })
    ret
  }
  
  // replace some multiple use VecDec with VecDecCopy so that it will be copied before being mutated
  def markCopy(lst: List[Let]) : List[Let] = {
    lst.foldRight[List[Let]](Nil)( (let, c) => let match {
      
      case VecLet(x, v) => { 
    	  def cnts(lst: List[Let], l: Let, v: Vec) = {
    		  lst.exists( let => let != l && let.contains(v) )  
    	  }
    	  // FIXME: This adds more "copies" than necessary to make parallelization safe 
    	  //        and cnts(c, v) would be more efficient for sequential programs.
        val ret = VecLet(x, v.subIfInplace(v1 => if(cnts(lst, let, v1)) new VecDecCopy(v1) else v1 )) 
        ret :: c
      }
      case _ => let::c
      
    })
  }
  
  // eliminate common sub-expressions
  def elim (lst: List[Let]): List[Let] = {
    var tail = lst
    var i = 0
    
    while (i < tail.size) {
      val let = tail(i)
      
    	let match {
          case VecLet(x, v) => {
            val eq = tail.filter({case VecLet(x1, v1) => x1 != x && v1 == v case _ => false})
            tail = tail.filter(x => !eq.contains(x))
            eq.foreach({case VecLet(x1, _) => tail = tail.map(l => l.sub(x1, x)) 
                        case _ =>})
          } 
          case MatLet(x, m) => {
            val eq = tail.filter({case MatLet(x1, m1) => x1 != x && m1 == m case _ => false})
            tail = tail.filter(x => !eq.contains(x))
            eq.foreach({case MatLet(x1, _) => tail = tail.map(l => l.sub(x1, x))
                        case _ =>})
          }
          case _ =>  
        }
      
      i = i + 1
    }
    tail
  }
  
  import deepdsl.gradient.IR._
  
  type Env = List[Let]
   
//  def apply(lst: Env) : Env = {
//     lst.foldLeft(unit[Env, Unit](()))((c, e) => c >> (apply1(e) >>= (l => modify(s => s :+ l)))).execState(Nil)
//  }
  
  def apply(let: Let) : Env = { val (l, s) = apply1(let).runState(Nil); s :+ l }
  
  def apply1(let: Let) : MonadState[Env, Let] = {
    let match {
      case Update(key, v) => unit(let)
      case ExpLet(y, e) => { 
        apply1(e) >>= (x => unit(ExpLet(y, x)))
      } 
      case VecLet(x, v) => { 
        apply2(v) >>= (x1 => unit(VecLet(x, x1)))
      } 
    }
  }
  
  def apply1(m: Mat) : MonadState[Env, Mat] = {
    m match {
      case MatOfVec(i, v) => { 
        apply1(v) >>= (x => unit(MatOfVec(i, x)))
      }
      case _ => {
        val x = Mat._new(m) 
        modify((s: Env) => s :+ MatLet(x, m)) >> unit(x)
      }
    }
  }
  
  def getVecVar(v: Vec) = T._new(v.getDims)
  
  private def apply1(e: AExp) : MonadState[Env, AExp] = {
    e match {
        case ScalarDec(_) => unit(e)
        case Accuracy(p, y, k) => { 
          apply1(p) >>= (x1 => unit(Accuracy(x1, y, k)))
        }
        case VecElem(_, _) | Num(_) | IndexExp(_) | DimSize(_) | Real(_,_) => unit(e)
        case InnerProd(v1, v2) => { 
          apply1(v1) >>= (x1 => apply1(v2) >>= (x2 => unit(InnerProd(x1, x2))))
        }
        case Plus(e1, e2) => { 
          apply1(e1) >>= (x1 => apply1(e2) >>= (x2 => unit(Plus(x1, x2))))
        }
        case Times(e1, e2) =>  { 
        	apply1(e1) >>= (x1 => apply1(e2) >>= (x2 => unit(Times(x1, x2))))
        } 
        case Delta(il, ir, e1) => { 
          apply1(e1) >>= (x1 => unit(Delta(il, ir, x1)))
        }
        case Log(e1) => { 
          apply1(e1) >>= (x1 => unit(Log(x1)))
        }
        case Exp(e1) => { 
          apply1(e1) >>= (x1 => unit(Exp(x1)))
        }
        case Pow(e1, x) => { 
          apply1(e1) >>= (x1 => unit(Pow(x1, x)))
        }
        case Max(i, e1) => { 
          apply1(e1) >>= (x1 => unit(Max(i, x1)))
        }
        case ScalarApp(fun, arg) =>  { 
          apply1(arg) >>= (x1 => unit(ScalarApp(fun, x1)))
        }
        case Vec2ScalarApp(fun, arg) =>  { 
          apply1(arg) >>= (x1 => apply1(fun.body.sub(fun.param, x1)))
        }
        case Sum(i, e1) => { 
          apply1(e1) >>= (x1 => unit(Sum(i, x1)))
        }
        case MaxVec(v1) => { 
          apply1(v1) >>= (x1 => unit(MaxVec(x1)))
        }
        case Indicator(e1, e2) => { 
          apply1(e1) >>= (x1 => apply1(e2) >>= (x2 => unit(Indicator(x1, x2))))
        }
        case SumVec(v1) => {
          apply1(v1) >>= (x1 => unit(SumVec(x1)))
        } 
    }
  }
  
  private def apply1(v: Vec) : MonadState[Env, Vec] = {  
      apply2(v) >>= (v2 => v2 match {
          case _: VecAsIndicator 
          | _: CudaVec => {  
            val x = getVecVar(v2) 
            modify((s: Env) => s :+ VecLet(x, v2)) >> unit(x)
            // We need to maintain the right order of execution
          }
          case VecDec(_,_)
          | FlattenVec(_, _, _) 
          | UnFlattenVec(_, _, _) 
          | VecSlice(_, _) 
          | VecApp(_,_) => unit(v2)
          
          case _ => {
            val x = getVecVar(v2)
            modify((s: Env) => s :+ VecLet(x, v2)) >> unit(x)
            // We need to maintain the right order of execution
          }
        }
      )
  } 
  
  private def apply2(v: Vec) : MonadState[Env, Vec] = {
      v match { 
        case VecDec(_,_) 
        | FlattenVec(_, _, _) 
        | UnFlattenVec(_, _, _) 
        | VecSlice(VecDec(_, _), _) => unit(v)
        
        case VecSlice(v1, i) => apply1(v1) >>= (x1 => unit(VecSlice(x1, i)))  
       
        case VecFlexSlice(v1, _) if (v1.isInstanceOf[VecDec]) => unit(v)
        case VecFlexSlice(v1, pattern) => apply1(v1) >>= (x1 => unit(VecFlexSlice(x1, pattern)))
        
        case VecClip(v1, b) => apply1(v1) >>= (x1 => unit(VecClip(x1, b)))
        
        case VecExp(i, e) =>  { 
          if (i.exists(j => e.free(j))) { 
            unit(v)
          }
          else {
            apply1(e) >>= (e1 => unit(VecExp(i, e1))) 
          }
        }
        case VecProd(m1, m2) => {  
          apply1(m1) >>= (x1 => apply1(m2) >>= (x2 => unit(VecProd(x1, x2))))
        }
        case PointwiseProd(v1, v2) => {
          apply1(v1) >>= (x1 => apply1(v2) >>= (x2 => unit(PointwiseProd(x1, x2)))) 
        }
        case VecApp(fun, arg) => { 
          apply1(arg) >>= (x1 => apply1(fun.body.sub(fun.param, x1)) )
        }
        case ScalarVecApp(fun, arg) => { 
          apply1(arg) >>= (x1 => unit(ScalarVecApp(fun, x1)))
        }
        case VecPlus(v1, v2) => { 
          apply1(v1) >>= (x1 => apply1(v2) >>= (x2 => unit(VecPlus(x1, x2))))
        }
        case ExpVec(v1) => { 
          apply1(v1) >>= (x1 => unit(ExpVec(x1)))
        }
        case LogVec(v1) => { 
          apply1(v1) >>= (x1 => unit(LogVec(x1)))
        }
        case VecTimesScalar(v1, e) => { 
          apply1(v1) >>= (x1 => unit(VecTimesScalar(x1, e)))
        }
        case IndicatorVec(v1, v2) => { 
          apply1(v1) >>= (x1 => apply1(v2) >>= (x2 => unit(IndicatorVec(x1, x2))))
        }
        case MaxMat(m) => { 
          apply1(m) >>= (x1 => unit(MaxMat(x1)))
        }
        case SumMat(m) => { 
          apply1(m) >>= (x1 => unit(SumMat(x1)))
        }
        case CopyVec(i, v1) => {    
          apply1(v1) >>= (x1 => unit(CopyVec(i, x1)))
        }
        case PowVec(v1, e) => { 
          apply1(v1) >>= (x1 => unit(PowVec(x1, e)))
        }
        case FixVec(name,p,dim) => { 
          if(!isTraining && name.isInstanceOf[Dropout]) {
            apply1(p) >>= (x1 => unit(x1(0)))
          }
          else {
            apply1(p) >>= (x1 => unit(FixVec(name, x1, dim)))
          }
        }
        case FixProd(v1, FixGrad(FixVec(name, p, dim), activeP, dx)) => {         
          apply1(v1) >>= (x1 => apply1(p) >>= (x2 => apply1(activeP) >>= (x3 => 
            unit(FixProd(x1, FixGrad(FixVec(name, x2, dim), x3, dx))))))
        }
        case ConcatVec(lst) => {
        	apply1(lst) >>= (xlst => unit(ConcatVec(xlst)))
        }
        case ProjVec(dy, lst, n) => { 
          apply1(lst) >>= (x1 => apply1(dy) >>= (x2 => unit(ProjVec(x2, x1, n))))
        }
        case _ => unit(v)
      }
  } 
  def apply1(lst: List[Vec]) : MonadState[Env, List[Vec]] = 
    lst.foldRight(unit[Env, List[Vec]](Nil))((e, c) => c >>= (l => apply1(e) >>= (v => unit(v::l))))
}

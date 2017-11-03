package deepdsl.optimization

import deepdsl.derivation._
import deepdsl.run._
import deepdsl.layer._
import deepdsl.lang._
import scala.Range
import deepdsl.ast._

object Optimize {
  def apply(f: Vec2ScalarFun) : Vec2ScalarFun = (code_motion(loop_merging(f)))
  def apply(f: VecFun) : VecFun = (code_motion(loop_merging(f)))
  def apply(v: Vec) : Vec = (code_motion(loop_merging(v)))   
  def apply(e: AExp) : AExp = (code_motion(loop_merging(e)))
   
  def apply(let: Let) : Let = let match {
    case Update(key, v) => Update(key, Optimize(v))
    case ExpLet(x, e) => ExpLet(x, Optimize(e)) 
    case VecLet(x, v) => VecLet(x, Optimize(v))
    case _ => let
  } 
}

object vectorize {
    // shift + i or i + shift 
    // shift - i or
    // i / sampling-rate
  def isEqualOrShiftOrUpSample(idx: Index, dim: Dim, id: IndexDec) = {  
  
    val r = (idx == id && dim == id.dim) || (idx match {
      case IndexOpIndex(left, PlusOp, right) => (left == id || right == id)
      case QuotientRemainder(_, _, r) => r == id 
      case IndexOpIndex(_, MinusOp, right) => right == id 
      case IndexOpConst(left, DivOp, d) => (left == id && dim == id.dim / d)   // TODO: here we made sure that we only up-sample the exact same index
      case _ => false
    })
    r
  }
  def getPattern(idx: Index, i: List[IndexDec]) = {
    var ret: Pattern = null
    
    idx match {
      case IndexOpIndex(left, PlusOp, right)  => for(id <- i if ret == null) { if (left == id) ret = ShiftPattern(right, id) else if (right == id) ret = ShiftPattern(left, id) } 
      case QuotientRemainder(q, d, r)         => for(id <- i if ret == null) { if (r == id) ret = ShiftPattern(q * d, id) } 
      case IndexOpIndex(left, MinusOp, right) => for(id <- i if ret == null) { if (right == id) ret = ShiftFlipPattern(left, id) } 
      case IndexOpConst(left, DivOp, right)   => for(id <- i if ret == null) { if (left == id) ret = UpSamplePattern(right, id) } 
      case IndexDec(_,_)                      => for(id <- i if ret == null) { if (idx == id) ret = IndexPattern(id) }
    }
    
    if (ret == null) ret = SlicePattern(idx)
    
    ret
  }

  private def isUnFlatten(i: List[IndexDec], e: VecElem): (Boolean, (Int, List[Dim])) = {
    val j = e.idx; val v = e.vec
    
    for(x <- 0 to j.size - 1) {
      if (i(x) != j(x)) {
        val lst = dim2DimList(v.dim(x))
        if (lst.size > 1 && 
            j(x).unflatten(lst) == i.slice(x, x+lst.size)) {
          if (i.size == j.size + lst.size - 1 && Range(x+1, j.size).forall(y=>j(y) == i(y+lst.size-1)))
            return (true, (x, lst))
        }
      }
    }
    return (false, (0, List()))
  }
  
  private def isFlatten(i: List[IndexDec], j: List[Index]): (Boolean, (Int, List[Dim])) = {
    for(x <- 0 to i.size - 1) {
      if (i(x) != j(x)) {
        val lst = dim2DimList(i(x).dim)
        if (lst.size > 1 && i(x).unflatten(lst) == j.slice(x, x+lst.size)) {
          if (i.size + lst.size - 1 == j.size && Range(x+1, i.size).forall(y=>i(y) == j(y+lst.size-1))) 
            return (true, (x, lst))
        }
      }
    }
    return (false, (0, List()))
  }
  
  private def dim2DimList (d: Dim) : List[Dim] = {
    d match {
      case DimOpConst(d1, TimesOp, DimSize(d2)) => dim2DimList(d1) ++ List(d2)
      case d1 @ DimOpConst(_,_,_) => List(d1)
      case d1 @ DimConst(_) => List(d1)
      case d1 @ DimVar(_) => List(d1)
      case _ => Nil
    }
  }
  
  def getMat(i: List[IndexDec], v: Vec): Mat = {
          val ret = MatExp(i, v)
          
          v match {
            case VecSlice(vec, slices) => 
              if (i.size == slices.size && i.zip(slices).zip(vec.getDims).forall({case ((i, s), d) => i == s && i.dim == d}))
                MatOfVec(i, vec)
              else
                ret  
            case vec @ VecFlexSlice(VecDec(_, d), p) => {
              val si = p.foldLeft[List[Index]](Nil)((c,e) => e match { 
                                                              case IndexPattern(_)=> c  
                                                              case SlicePattern(j) => c ++ List(j)  
                                                              case ShiftPattern(j, i) => c ++ List(j) 
                                                              case ShiftFlipPattern(j, i) => c ++ List(j)  
                                                              case UpSamplePattern(_,_) => c })
                
              if (si.zip(i).forall({case (x, y)=> x.contains(y)})) 
                MatOfVecFlexSlice(i, vec)
              else 
                ret
            }
            case _ => ret
          }
  }
  
  def apply(vec: VecExp) : Vec = {
    val i = vec.indices; val e = vec.e
    
    e match {
      case InnerProd(VecExp(i1, e1), VecExp(i2, e2)) => SumMat(MatOfVec(i, PointwiseProd(VecExp(i++i1, e1), VecExp(i++i2, e2))))
      case InnerProd(VecSlice(v1, i1), VecSlice(v2, i2)) if(i == i1 && i == i2) => SumMat(MatOfVec(i, PointwiseProd(v1, v2)))
      case InnerProd(v1, v2) => vectorize(VecExp(i, SumVec(PointwiseProd(v1, v2))))
      
      case e @ VecElem(v, j) => { 
        if (i.size > j.size) {
          val (b, (x, d)) = isUnFlatten(i, e)
          if (b) 
            UnFlattenVec(v, x, d) 
          else if (i.drop(i.size-j.size).zip(j).forall({case (a,b) => a == b})) {
            CopyVec(i.take(i.size-j.size), v)  
          }
          else 
            vec 
        }
        else if (i.zip(j.drop(j.size-i.size)).forall({case (a,b) => a == b})) {  
          if (i.size == j.size)                   // (i1,..,in) => V[i1,..,in]          --> V
            v 
          else                                    // (i1,..,in) => V[j1,..,jm,i1,..,in] --> slice V by j1..jm
            v.of(j.take(j.size-i.size):_*)                  
        }
        else if (i.forall(idx => (j zip v.dim).exists({case (idx2, d) => isEqualOrShiftOrUpSample(idx2, d, idx)}))) {
          val vfs = VecFlexSlice(v, j.map (index => getPattern(index, i))) 
          if (vfs.getIndices == i) // MUST preserve the order of "i" for this vector slice to be legal
            vfs 
          else  
            vec 
        }  
        else {
          val (b, (x, d)) = isFlatten(i, j)
          if (b) {
            FlattenVec(v, x, d)
          }
          else {
            vec
          }
        }
      }
      // refactor up-sampling out of max-pooling vector to enable common subexpression elimination for the max-pooling vector 
      case MaxVec(vs @ VecFlexSlice(v, pattern)) => {
        val (i1, p1, p2) =
        (i zip pattern).map( {
          case (idx, p@SlicePattern(j)) => if (idx == j) (idx, p, IndexPattern(idx)) else (idx, p, null)
          case (idx, p@ShiftPattern(j, k)) => {
            j.getUpSample(idx) match {
              case Some(d) => { val idx1 = T.index(idx.dim/d); 
              (idx1, ShiftPattern(j.sub(idx, idx1*d), k), UpSamplePattern(d, idx)) }
              case None => (idx, p, null)
            }
          }
        } ).unzip3
        if (! p2.contains(null)) 
          VecFlexSlice(MaxMat(getMat(i1, VecFlexSlice(v, p1))), p2) 
        else 
          MaxMat(getMat(i, vs))
      }
      
      case MaxVec(VecSlice(v, j)) if(i==j) => MaxMat(MatOfVec(i, v))
      
      case _  if (i.exists(idx => ! e.free(idx)) && i.exists(idx => e.free(idx))) => { 
        	val innerIdx = i.filter(idx => e.free(idx))
        	val v = T._new(innerIdx.size)
          VecApp(VecFun(v, vectorize(VecExp(i, v(innerIdx:_*)))), vectorize(VecExp(innerIdx, e)))
        }
    

//      case SumVec(v) => SumMat(getMat(i, v))
      
      case SumVec(v: VecSlice) => SumMat(getMat(i, v))
      case SumVec(v: VecFlexSlice) => SumMat(getMat(i, v)) 
      case SumVec(v) => {
        val i1 = i.filter(idx => v.free(idx))
        if(i1.size == i.size) {
          val j = v.getIndices
          val x = code_motion(VecExp(i++j, v.get(j)))
          
          SumMat(getMat(i, x.of(i:_*)))
        }
        else if(i1.size == 0) {
          VecExp(i, SumVec(code_motion(v)))
        }
        else { 
          val j = v.getIndices
          val x = code_motion(VecExp(i1++j, v.get(j)))
          val y = SumMat(getMat(i1, x.of(i1:_*)))
          VecExp(i, y.get(i1))
        }
      }

      case _ => vec  
          
    }
  }
}

// move loop invariant out of the loop
object code_motion {
  def apply(f: Vec2ScalarFun): Vec2ScalarFun = Vec2ScalarFun(f.param, code_motion(f.body))
  def apply(f: VecFun): VecFun = VecFun(f.param, code_motion(f.body))
  
  def apply(v: Vec) : Vec = v match {
    case PointwiseProd(v1, v2) => PointwiseProd(code_motion(v1), code_motion(v2)) 
      
    case VecExp(i0, e) => { 
      val ex = code_motion(e)
//      val i1 = i0.filter(j => ex.free(j))
//      val i = if (i1.size > 0) i1 else i0
      val i = i0
      val vec = ex match {
        case ScalarApp(fun, arg) => code_motion( ScalarVecApp(fun, code_motion(VecExp(i, arg))) )
        case x @ Vec2ScalarApp(fun, arg) => 
          if (! i.exists(idx => arg.free(idx))) 
            VecApp(VecFun(fun.param, VecExp(i, fun.body)), arg)
          else
            VecExp(i, x) 
        case Log(e1) => LogVec(code_motion(VecExp(i, e1)))
        case Exp(e1) => ExpVec(code_motion(VecExp(i, e1)))
        case Plus(e1, e2) => VecPlus(code_motion(VecExp(i, e1)), code_motion(VecExp(i, e2)))
        case x @ Times(e1, e2) => 
          if (! i.exists(idx => e2.free(idx)))
            VecTimesScalar(code_motion(VecExp(i, e1)), e2)
          else if (! i.exists(idx => e1.free(idx)))
            VecTimesScalar(code_motion(VecExp(i, e2)), e1)
          else // if (i.forall(idx => e2.free(idx)) && i.forall(idx => e1.free(idx))) 
            PointwiseProd(code_motion(VecExp(i, e1)), code_motion(VecExp(i, e2)))
          // else VecExp(i, x)
        case Max(j, e1) => code_motion(VecExp(i, MaxVec(code_motion(VecExp(j.toList, e1))))) 
            
        case Indicator(e1, e2) => IndicatorVec(code_motion(VecExp(i, e1)), code_motion(VecExp(i, e2)))
        
        case InnerProd(v1, v2) => {       
          val (j1, j2) = (i.filter(idx=> v1.free(idx)), i.filter(idx=> v2.free(idx)))  
          
          if (i.size == j1.size + j2.size) {  // j1 and j2 must not share any indices
          	val m1 = vectorize.getMat(j1, v1)
            val m2 = vectorize.getMat(j2, v2)
          	val (p, j) = if (j1.size > 0 && j2.size > 0 && i.last == j1.last) (VecProd(m2, m1), j2++j1) else (VecProd(m1, m2), j1++j2)
          	
            if (i == j)                // p's indices must be in the same order as i
              p
            else { 
            	val x = T._new(p.size) 
            	VecApp(VecFun(x, VecExp(i, x.get(j))), p)
            }
          }
          else 
            vectorize(VecExp(i, InnerProd(v1, v2)))  
        }
        
        case Pow(e1, x) => PowVec(code_motion(VecExp(i, e1)), x)
        
        case x @ _ => vectorize(VecExp(i, x)) 
      } 
      if(i0.size == i.size || i.size == 0) vec else {
        val x = T._new(i.map(j=>j.dim))
        VecFun(x, VecExp(i0, x.get(i)))(vec)
      }
    }
    case VecSlice(v, i) => VecSlice(code_motion(v), i)
    case VecApp(fun, arg) => VecApp(code_motion(fun), code_motion(arg))
    
    case ScalarVecApp(ScalarFun(param, ScalarApp(fun, arg1)), arg2) => {
      if (param == arg1) 
        ScalarVecApp(fun, arg2)
      else
        code_motion( ScalarVecApp(fun, ScalarVecApp(ScalarFun(param, arg1), arg2)) )
    }
    case ScalarVecApp(fun, arg) => ScalarVecApp(fun, code_motion(arg))
    case ExpVec(VecSlice(v, i)) => VecSlice(ExpVec(code_motion(v)), i)
    case ExpVec(v) => ExpVec(code_motion(v))
   
    case IndicatorVec(v1, v2) => IndicatorVec(code_motion(v1), code_motion(v2))
    
    case VecClip(v1, b) => VecClip(code_motion(v1), b)
    
    case VecPlus(v1, v2) => refactor(VecPlus(refactor(code_motion(v1)), refactor(code_motion(v2))))
    
    case VecTimesScalar(v1, e) => VecTimesScalar(code_motion(v1), e)
    
    case _ => v  // TODO: why not going into more cases ? 
  }
  
  def refactor(v: Vec) : Vec = v match {
    case VecPlus(PointwiseProd(v11, v12), PointwiseProd(v21, v22)) if (v11 == v21) => PointwiseProd(v11, refactor(VecPlus(v12, v22)))
    case VecPlus(PointwiseProd(v11, v12), PointwiseProd(v21, v22)) if (v12 == v22) => PointwiseProd(v12, refactor(VecPlus(v11, v21)))
    case VecPlus(PointwiseProd(v11, v12), PointwiseProd(v21, v22)) if (v11 == v22) => PointwiseProd(v11, refactor(VecPlus(v12, v21)))
    case VecPlus(PointwiseProd(v11, v12), PointwiseProd(v21, v22)) if (v12 == v21) => PointwiseProd(v12, refactor(VecPlus(v11, v22)))
    case _ => v
  }
  
  def apply(e: AExp) : AExp = e match {
    case InnerProd(v1, v2) => {
//        val replacePattern = (i: IndexDec, p1: List[Pattern], p2: Pattern) => {
//          p1.foldRight[List[Pattern]](Nil)((e, c) => e match {
//            case IndexPattern(j) => if (j.dim == i.dim) p2::c else e::c    // TODO: should compare j with i but can't because of object identity
//            case _ => e::c                                                 // FIXME: This is now broken since j.dim and i.dim are constants 
//          })
//        }
//        
//        val switchPattern = (v1: VecFlexSlice, v2: VecFlexSlice) => {
//          val (p1, p2) = (v2.p.zip(v2.v.getIndices)).foldRight[(List[Pattern], List[Pattern])]((v1.p, Nil))((e, c) => e match {
//              // TODO: This is too specific and needs to be relaxed to be more applicable
//              case (ShiftFlipPattern(s, j2), k) => (replacePattern(j2, c._1, ShiftFlipPattern(s, k)), IndexPattern(k)::c._2) 
//              case _ => (c._1,  e._1::c._2)
//          })
//          (VecFlexSlice(v1.v, p1), VecFlexSlice(v2.v, p2))
//        }
        
        val (v11, v12) = (code_motion(v1), code_motion(v2))  
//        match {
//          case (x@VecSlice(_, _), y@VecFlexSlice(_, p)) => if (p.exists({case ShiftFlipPattern(_,_)=>true case _ =>false})) switchPattern(x.toFlexSlice, y) else (x,y)
//          case (x@VecFlexSlice(_, _), y@VecFlexSlice(_, p)) => if (p.exists({case ShiftFlipPattern(_,_)=>true case _ =>false})) switchPattern(x, y) else (x,y)
//          case x@_ => x
//        }
        
        // We went through all the above trouble so that Y[j] * W[i-j] is converted to Y[i-k] * W[k]
        // strictly speaking, this may not benefit if k.dim is not less than j.dim
        // TODO: This is abandoned since it is not only broken but also slower for LeNet. 
        
        InnerProd(v11, v12)
    }
    case Plus(e1, e2) => Plus(code_motion(e1), code_motion(e2)) 
    case Times(e1, e2) => Times(code_motion(e1), code_motion(e2))  
    case Delta(il, ir, e1) => Delta(il, ir, code_motion(e1))
    case Log(e) => Log(code_motion(e))
    case Exp(e) => Exp(code_motion(e))
    case Pow(e, x) => Pow(code_motion(e), x)
    case Max(i, v) => Max(i, code_motion(v))
    case ScalarApp(fun, arg) =>  ScalarApp(fun, code_motion(arg)) 
    case Vec2ScalarApp(fun, arg) =>  Vec2ScalarApp(code_motion(fun), code_motion(arg))
    case Sum(i, e) => Sum(i, code_motion(e))
    case SumVec(v) => SumVec(code_motion(v))
    case MaxVec(v) => MaxVec(code_motion(v))
    case Indicator(e1, e2) => Indicator(code_motion(e1), code_motion(e2))
    case VecElem(_, _) | DimSize(_) | Num(_) | Real(_,_) => e
    case Accuracy(p, y, k) => Accuracy(code_motion(p), y, k)
    case _ => e
  }
}

// 1. convert sum of product to inner product
// 2. convert vector of product to point-wise product
object loop_merging {
  def apply(f: Vec2ScalarFun): Vec2ScalarFun = Vec2ScalarFun(f.param, loop_merging(f.body))
  def apply(f: VecFun): VecFun = VecFun(f.param, loop_merging(f.body))
  
  def apply(v: Vec): Vec = 
    v match {
      case VecExp(i, e) => loop_merging(e) match {
        case x @Times(e1, e2) => 
          if (i.forall(idx => e1.free(idx) && e2.free(idx))) 
            PointwiseProd(VecExp(i, e1), VecExp(i, e2)) 
          else 
            VecExp(i, x) 
        case x @ _ => VecExp(i, x)
      }
      case VecSlice(v, i) => VecSlice(loop_merging(v), i)
      case VecApp(fun, arg) => VecApp(loop_merging(fun), loop_merging(arg))
      case ScalarVecApp(fun, arg) => ScalarVecApp(fun, loop_merging(arg))
      case VecClip(v, b) => VecClip(loop_merging(v), b)
      case VecPlus(v1, v2) => VecPlus(loop_merging(v1), loop_merging(v2))
      case VecTimesScalar(v1, e) => VecTimesScalar(loop_merging(v1), e)
      case _ => v
    }
  
  def apply(e: AExp): AExp =  
    e match { 
      case Sum(i, e1) => {
        val x = loop_merging(e1) 
        val ret = Sum(i, x)
        x match {  
          case Times(e11, e12) if(e11.free(i) && e12.free(i)) => InnerProd(VecExp(List(i), e11), VecExp(List(i), e12)) 
          case InnerProd(VecExp(i1, e11), VecExp(i2, e12)) if(e11.free(i) && e12.free(i))  =>  InnerProd(VecExp(i::i1, e11), VecExp(i::i2, e12))  
          case VecElem(v, idx) if (idx.exists(index => index.contains(i))) => SumVec(VecExp(List(i), x))  
          case y@Exp(VecElem(v, idx)) if (idx.exists(index => index.contains(i))) => SumVec(ExpVec(VecExp(List(i), y.e)))  
          case SumVec(VecExp(indices, VecElem(v, idx))) if (idx.exists(index => index.contains(i))) => SumVec(VecExp(i::indices, VecElem(v, idx)))  
          case Indicator(e1, e2) if(e1.free(i) && e2.free(i)) => SumVec(IndicatorVec(VecExp(List(i), e1), VecExp(List(i), e2))) 
          
          case SumVec(VecExp(j, e)) => SumVec(VecExp(i::j, e))
//          case _ if !x.free(i) => x * i.dim.size 
          case _ => SumVec(VecExp(List(i), x))
//          case _ => ret
        }
      }  
    	case Plus(e1, e2) => Plus(loop_merging(e1), loop_merging(e2)) 
    	case Times(e1, e2) => Times(loop_merging(e1), loop_merging(e2))  
    	case Delta(il, ir, e1) => Delta(il, ir, loop_merging(e1))
    	case Log(e) => Log(loop_merging(e))
    	case Exp(e) => Exp(loop_merging(e))
    	case Pow(e, x) => Pow(loop_merging(e), x)
    	case Max(i, v) => {
    	  val x = loop_merging(v)
    	  val ret = Max(i, x)
    	  x match {
    	    case VecElem(v, idx) if(i.forall(j=>idx.exists(index => index.contains(j)))) => MaxVec(VecExp(i.toList, x))
    	    case _ => ret
    	  }
    	}
    	case ScalarApp(fun, arg) =>  ScalarApp(fun, loop_merging(arg)) 
    	case Vec2ScalarApp(fun, arg) =>  Vec2ScalarApp(loop_merging(fun), loop_merging(arg))
    	case Indicator(e1, e2) => Indicator(loop_merging(e1), loop_merging(e2))
    	case Accuracy(p, y, k) => Accuracy(loop_merging(p), y, k)
      case _ => e
    }
}

 
package deepdsl.ast

import deepdsl.analysis.TypeException
import deepdsl.layer._
import deepdsl.optimization._
import deepdsl.gradient.Gradient
import deepdsl.lang._ 
import deepdsl.derivation._ 

case class VecGradient (fun: VecDec => Vec) {
  def *(that: VecGradient) = VecGradient(x => fun(x) * that)
  def apply(dy: VecDec) = fun(dy) 
}

trait Vec extends VecOp {     
	def +(v: Vec) =  VecPlus(this, v)
  def *(e: AExp) = e match {
	  case Num(1) => this
	  case _ => VecTimesScalar(this, e)
	}
  
  // inner product between this and v: not defined if v does not have indices of the same length
  def *(v: Vec) : AExp = { 
    if (size != v.size) 
      throw TypeException(s"$this and $v have different sizes")
    
    // create a list of fresh indices
    val i = getDims.map(d => T.index(d)) 
    // for all i1 .. i_n. sum(this[i1, .., i_n] * v[i1, .., i_n])
    val ret = i.foldRight[AExp](this.get(i) * v.get(i))((idx, e) =>  Sum(idx, e))
    Simplify(ret)
  }
  
  def *(v: VecGradient) : Vec =  { 
     val x = T._new(getDims)
     VecFun(x, v(x))(this)  
  }  
  
  // this * w with zero padding and step 1
  def convolv(w: Vec) = {
    val k = w.getDims.map (d => T.index(d)) 
    val D = getDims.zip(w.getDims).map( p => p._1-p._2+Num(1) )
    val j = D.map(d => T.index(d))
    val i = j.zip(k).map ({case (j1, k1) => j1 + k1}) // TODO: i + j instead of i + j - 1 since indices start from 0.
    
    VecExp(j, k.foldRight[AExp](this.get(i) * w.get(k))((idx, e) => Sum(idx, e)))
  }
  
	def clip(bound: AExp) = VecClip(this, bound)
	 
  def size = getDims.size
  
  def apply (i: Index*) = get(i.toList) 
  def of(i: Index*) = this match {
	  case VecSlice(v, slice) => VecSlice(v, slice++i)    
	  case _ => VecSlice(this, i.toList)
	}
  
  def grad(x: VecDec) : VecGradient = Gradient(this, x)
} 

//this can be train or test data, dataType can be image or label
case class VecData (dataset: Data, option: (Boolean, String)) extends Vec 
//case class VecInit (init: Init, path: String, dim: List[Dim]) extends Vec

case class VecVar (name: String) {
  override def toString = name
  override def equals(that: Any) = super.equals(that)
}

case class VecDec (v: VecVar, dim: List[Dim]) extends Vec { 
  def <-- (rhs: Vec) = VecLet(this, rhs)
  
//  override def apply(i: Index*) = get(i.toList)  
  override def sub(dl: Dim, dr: Dim): VecDec = this match {
    case x:VecDecCopy => new VecDecCopy(x.dec.sub(dl, dr))
    case x:VecParam => new VecParam(x.init.sub(dl, dr), x.v, x.dim.map(d => d.sub(dl, dr))) 
    case x:VecAsIndicator => new VecAsIndicator(x.vec.sub(dl, dr), x.numOfCls)
    case x:CudaVec => new CudaVec(x.vec.sub(dl, dr)) 
    case _  => VecDec(v, dim.map(dx => dx.sub(dl, dr))) 
  }
  def asCuda = new CudaVec(this)
  def asIndicator(k: Int): VecAsIndicator = asIndicator(DimConst(k))
  def asIndicator(K: Dim) = new VecAsIndicator(this, K)
}

class VecDecCopy(val dec: VecDec) extends VecDec(dec.v, dec.dim) {
  // FIXME: This class is a temporary solution, applicable to SSA code only and we know for sure "v" should be VecDec
  def this(v: Vec) = this(v.asInstanceOf[VecDec]) 
}
class VecParam (val init: Init, v: VecVar, dim: List[Dim]) extends VecDec(v, dim) {  
  def update(dv: Vec) = Update(this, dv)
}
// convert vecData to an indicator
class VecAsIndicator(val vec: VecDec, val numOfCls: Dim) extends VecDec(vec.v, vec.dim:::List(numOfCls)) 
// load vec to CUDA GPU
class CudaVec(val vec: VecDec) extends VecDec(vec.v, vec.dim)  

// keep values within (-bound, bound)
case class VecClip(v: Vec, bound: AExp) extends Vec  
case class VecSlice(v: Vec, slice: List[Index]) extends Vec {   
  def toFlexSlice = VecFlexSlice(v, slice.map(i=>SlicePattern(i))++getIndices.map(i=>IndexPattern(i)))
}
 
case class VecExp (indices: List[IndexDec], e: AExp) extends Vec {   
  override def equals(that: Any) = if (that.isInstanceOf[VecExp]) {
      val thatVec = that.asInstanceOf[VecExp]
      
      val b = size == thatVec.size && 
              indices.zip(thatVec.indices).forall({case(x,y)=>x.dim==y.dim}) && 
              e == thatVec.get(indices) 
      b
    }
    else 
      false  
      
  def + (e1: AExp) = VecExp(indices, e + e1)
}

case class UnFlattenVec(v: Vec, start: Int, flatDims: List[Dim]) extends Vec 
case class FlattenVec(v: Vec, start: Int, flatDims: List[Dim]) extends Vec  

// [e^v[0], e^v[1], ..., e^v[n]]
case class ExpVec(v: Vec) extends Vec
// [ln(v[0]), ln(v[1]), ..., ln(v[n])]
case class LogVec(v: Vec) extends Vec  
// [v[0]^x, v[1]^x, ..., v[n]^x]
case class PowVec(v: Vec, x: AExp) extends Vec 
// [v[0]*e, v[1]*e, ..., v[n]*e]
case class VecTimesScalar(v: Vec, e: AExp) extends Vec  
// [v1[0] + v2[0], v1[1] + v2[1], ..., v1[n] + v2[n]]
case class VecPlus(v1: Vec, v2: Vec) extends Vec 
// [1{v1[0]==v2[0], 1{v1[1]==v2[1], ..., 1{v1[n]==v2[n]}]
case class IndicatorVec(v1: Vec, v2: Vec) extends Vec  
// [v1[0] * v2[0], v1[1] * v2[1], ..., v1[n] * v2[n]]
case class PointwiseProd(v1: Vec, v2: Vec) extends Vec  
// make "i" copies of v 
case class CopyVec(i: List[IndexDec], v: Vec) extends Vec {   
  override def equals(that: Any) = if (that.isInstanceOf[CopyVec]) {
      val thatVec = that.asInstanceOf[CopyVec]
      size == thatVec.size && i.zip(thatVec.i).forall({case (x,y)=>x.dim == y.dim}) && v == thatVec.v
    }
    else
      false
}

case class MaxMat(m: Mat) extends Vec  

case class SumMat(m: Mat) extends Vec  

// (i1 => v1) * (i2 => v2) = (i1++i1) => v1 . v2
case class VecProd(m1: Mat, m2: Mat) extends Vec {  val indices = m1.i ++ m2.i  } 

case class VecFlexSlice(v: Vec, p: List[Pattern]) extends Vec { 
  def mixIndex(i: List[Index]) = { 
    val itr = i.iterator
    p.map({
      case  IndexPattern(_)  => itr.next
      case  SlicePattern(j) => j
      case  ShiftPattern(j, _)  => j + itr.next
      case  ShiftFlipPattern(j, _)  => j - itr.next
      case  UpSamplePattern(c, _)  => itr.next / c
    })
  }   
  
  override def equals(that: Any) = {
    if (that.isInstanceOf[VecFlexSlice]) {
      val thatSlice = that.asInstanceOf[VecFlexSlice]
      v == thatSlice.v && getIndices.zip(thatSlice.getIndices).forall({case (x,y)=>x.dim == y.dim})
    }
    else
      false
  }
}
// f_v v2
case class VecApp (fun: VecFun, arg: Vec) extends Vec  
// f_e v => v'
case class ScalarVecApp(fun: ScalarFun, arg: Vec) extends Vec  
// lst[0] + lst[1] + ... + lst[n] where + means concatenation by channel axis (in NCHW format)
case class ConcatVec(lst: List[Vec]) extends Vec 

// This is to project the gradient of the concatenation tensor. n: index of the projected tensor
case class ProjVec(dy: Vec, lst: List[Vec], n: Int) extends Vec  
// parameters: input (x), actual parameters (e.g. weight and bias, if any) 
case class FixVec(layer: CudaLayer, param: List[Vec], dim: List[Dim]) extends Vec {  
  def _sub(vl: Vec, vr: Vec) = FixVec(layer, param.map(p=>p.sub(vl, vr)), dim)
  def _sub(dl: Dim, dr: Dim) = FixVec(layer, param.map(p=>p.sub(dl, dr)), dim.map(d=>d.sub(dl, dr)))
}
// v * dy/dx
case class FixProd(v: Vec, g: FixGrad) extends Vec  

// dy / dx
// all parameters: input (x), layer parameters (e.g. weight and bias, if any) 
// active parameters: the actual parameters used (in the order of output (y), input (x), and layer parameters)
case class FixGrad(v: FixVec, activeParam: List[Vec], x: Vec) {
  def sub(vl: Vec, vr: Vec) = FixGrad(v._sub(vl, vr), activeParam.map(p=>p.sub(vl, vr)), x.sub(vl, vr))
  def contains(v: Vec) = activeParam.contains(v)  
  def sub(dl: Dim, dr: Dim) = FixGrad(v._sub(dl, dr), activeParam.map(p=>p.sub(dl, dr)), x.sub(dl, dr))
  override def toString = s"d_${v.layer}(${activeParam.mkString(",")})/d_$x"
}


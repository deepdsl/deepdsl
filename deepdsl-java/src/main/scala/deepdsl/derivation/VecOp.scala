package deepdsl.derivation

import deepdsl.layer._
import deepdsl.analysis.TypeException
import deepdsl.lang._
import deepdsl.ast._

trait VecOp {
  override def toString = this match {
//    case VecInit(init, _, _) => init.toString
    case VecData(_, option) => option._2
    case x:VecDecCopy => s"${x.dec}.copy"
    case x:VecAsIndicator => s"Indicator(${x.vec}, ${x.numOfCls})"
    case x:CudaVec => s"Cuda(${x.vec})"
    case VecDec(v, _) => v.toString
    case VecClip(v, bounds) => s"clip($v, $bounds)"
    case VecSlice(v, slice) => s"$v[${slice.mkString(",")}, ${List.range(0, v.size-slice.size).map(_=>"@").mkString(", ")}]"
    case VecExp(i, e) => s"(${i.mkString(", ")}) => $e"
    case PowVec(v, x) => if(x == Num(-1)) s"1/($v)"  else s"($v)^$x"
    case SumMat(m) => s"Sum($m)" 
    case VecProd(m1, m2) => s"$m1 * $m2"
    case MaxMat(m) => s"Max($m)"
    case CopyVec(i, v) => s"(${i.mkString(", ")}) => $v"
    case PointwiseProd(v1, v2) => s"$v1 .* $v2"
    case IndicatorVec(v1, v2) => s"1{$v1 == $v2}" 
    case VecPlus(VecTimesScalar(v11, Num(-1)), v2) => s"($v2 - $v11)"
    case VecPlus(v1, VecTimesScalar(v12, Num(-1))) => s"($v1 - $v12)"
    case VecPlus(v1, v2) => s"($v1 + $v2)"
    case VecTimesScalar(v, e) => e match {
      case Pow(e1, Num(-1)) => s"($v / $e1)"
      case Num(-1) => s"- $v"
      case _ => s"($v * $e)"
    }
    case LogVec(v) => s"Log $v"
    case ExpVec(v) => s"E^($v)"
    case FlattenVec(v, start, flatDims) => s"$v[$start><${start + flatDims.size - 1}]"
    case UnFlattenVec(v, start, flatDims) => s"$v[$start<>${start + flatDims.size - 1}]"
    case x@VecFlexSlice(v, p) => s"$v[${x.mixIndex(List.range(0, x.size).map(_=> T.dummy_index)).mkString(", ")}]"
    case ScalarVecApp(fun, arg) => s"($fun $arg)"
    case VecApp(fun, arg) => s"($fun $arg)"
    case ConcatVec(lst) => s"Concat(${lst.mkString(",")})"
    case ProjVec(dy, lst, n) => s"Proj($dy, ${lst.mkString(",")}, $n)"
    case FixVec(layer, param, dim) => s"$layer(${param.mkString(",")})"
    case FixProd(v, g) => s"$v * $g"
  }
  def get(i: List[Index]): AExp = this match {
    case x@VecDec(_, dim) => VecElem(x, i)
    case VecClip(v, _) => v.get(i)
    case VecSlice(v, slice) => v.get(slice++i)
    case VecExp(indices, e) => { // TODO: we should check dimensions of "i" against the dimensions of "indices"
      if (i.size != indices.size) 
        throw TypeException(s"$this has different number of indices than $i")
      
      var ret = e
      for(x <- 0 to i.size-1) {
        ret = ret.sub(indices(x), i(x))
      }
      ret
    }
    case UnFlattenVec(v, start, flatDims) => {
      val k = i.drop(start).take(flatDims.size)
      val f = k.zip(flatDims).tail.foldLeft(k.head)((c,e) => c * e._2.size + e._1)
      val j = i.take(start) ++ List(f) ++ i.drop(start + flatDims.size)
      v.get(j)
    }
    case FlattenVec(v, start, flatDims) => {
      val j = i.take(start) ++ i(start).unflatten(flatDims) ++ i.drop(start+1)
      v.get(j)
    } 
    case ExpVec(v) => Exp(v.get(i))
    case LogVec(v) => Log(v.get(i)) 
    case PowVec(v, x) => Pow(v.get(i), x)
    case VecTimesScalar(v, e) => v.get(i) * e 
    case VecPlus(v1, v2) => v1.get(i) + v2.get(i)
    case IndicatorVec(v1, v2) => Indicator(v1.get(i), v2.get(i))
    case PointwiseProd(v1, v2) => Times(v1.get(i), v2.get(i))
    case CopyVec(j, v) => v.get(i.drop(j.size))
    case MaxMat(m) => MaxVec((m.i zip i).foldLeft(m.vec)((c,e)=> c.sub(e._1, e._2))) 
    case x@VecProd(m1, m2) => VecExp(x.indices,  m1.vec * m2.vec).get(i)
    case SumMat(m) => VecExp(m.i, SumVec(m.vec)).get(i)
    case x@VecFlexSlice(v, _) => v.get(x.mixIndex(i)) 
    case VecApp(fun, arg) => Vec2ScalarFun(fun.param, fun.body.get(i))(arg)
    case ScalarVecApp(fun, arg) => fun(arg.get(i))
    case ConcatVec(_) => throw new RuntimeException("can't reference elements of concat vec")
    case FixVec(_,_,_) => throw new RuntimeException("can't reference elements of fix vec") 
    case FixProd(_,_) => throw new RuntimeException("can't reference elements of fix prod") 
  }
  
  def getDims: List[Dim] = this match { 
    case VecData(dataset, option) => if(option._2 == "image") dataset.dim.map(d => T.dim(d)) else List(T.dim(dataset.dim(0)))
//    case VecInit(_, _, dim) => dim
    case VecDec(_, dim) => dim 
    case VecClip(v, _) => v.getDims
    case VecSlice(v, slice) => v.getDims.slice(slice.size, v.size)
    case VecExp(i, _) => i.map(i => i.dim) 
    case PowVec(v, _) => v.getDims
    case SumMat(m) => m.i.map(j => j.dim)
    case x@VecProd(m1, m2) => x.indices.map(i => i.dim)
    case MaxMat(m) => m.i.map(j=>j.dim) 
    case CopyVec(i, v) => i.map(j => j.dim) ++ v.getDims
    case PointwiseProd(v1, _) => v1.getDims
    case IndicatorVec(v1, _) => v1.getDims  
    case VecPlus(v1, _) => v1.getDims
    case VecTimesScalar(v, _) => v.getDims
    case LogVec(v) => v.getDims
    case ExpVec(v) => v.getDims
    case FlattenVec(v, start, flatDims) => {
      val d = v.getDims
      d.take(start) ++ List(flatDims.reduceLeft(_*_)) ++ d.drop(start + flatDims.size)
    } 
    case UnFlattenVec(v, start, flatDims) =>{
      val d = v.getDims
      d.take(start) ++ flatDims ++ d.drop(start+1)
    } 
    case x@VecFlexSlice(v, p) => x.getIndices.map(i=> i.dim)
    case ScalarVecApp(_, arg) => arg.getDims
    case VecApp(fun, _) => fun.body.getDims 
    case x@ConcatVec(lst) => {
  		val dimsList = lst.map(l => l.getDims)
  	  val firstDims = dimsList(0)
  	  List(firstDims(0), dimsList.map(dims=>dims(1)).reduce((c,e) => c + e.size), firstDims(2), firstDims(2))
  	}  
    case ProjVec(_, lst, n) => lst(n).getDims
    case FixVec(layer, param, dim) => dim 
    case FixProd(_, g) => g.x.getDims
  }
  
  def getIndices : List[IndexDec] = this match { 
    case VecDec(_, dim) => dim.map(d => T.index(d))
    case VecClip(v, _) => v.getIndices
    case VecSlice(v, slice) => v.getIndices.slice(slice.size, v.size)
    case VecExp(i, _) => i
    case PowVec(v, x) => v.getIndices
    case SumMat(m) => m.i
    case x@VecProd(m1, m2) => x.indices
    case MaxMat(m) => m.i 
    case CopyVec(i, v) => i ++ v.getIndices
    case PointwiseProd(v1, _) => v1.getIndices
    case IndicatorVec(v1, _) => v1.getIndices  
    case VecPlus(v1, _) => v1.getIndices
    case VecTimesScalar(v, _) => v.getIndices
    case LogVec(v) => v.getIndices
    case ExpVec(v) => v.getIndices
    case FlattenVec(v, start, flatDims) => {
      val i = v.getIndices 
      val flatIndex = T.index(flatDims.reduceLeft(_*_))
      i.take(start) ++ List(flatIndex) ++ i.drop(start + flatDims.size)
    } 
    case UnFlattenVec(v, start, flatDims) => {
      val i = v.getIndices
      i.take(start) ++ flatDims.map(d => T.index(d)) ++ i.drop(start+1)
    } 
    case x@VecFlexSlice(v, p) => p.foldLeft[List[IndexDec]](Nil)( (c, e)=> e match { 
          case IndexPattern(i) => c ++ List(i) 
          case SlicePattern(_) => c
          case ShiftPattern(_, i) => c ++ List(i)
          case ShiftFlipPattern(_, i) => c ++ List(i)
          case UpSamplePattern(_, i) => c ++ List(i) }
      )
    case ScalarVecApp(_, arg) => arg.getIndices
    case VecApp(fun, _) => fun.body.getIndices
    case ConcatVec(lst) => getDims.map(d => T.index(d))
    case ProjVec(_, lst, n) => lst(n).getIndices
    case FixVec(layer, param, dim) => dim.map(d => T.index(d))
    case FixProd(_, g) => g.x.getIndices
  }
  
	def contains(m: MatDec): Boolean = {
    this match {
      case VecPlus(v1, v2) => v1.contains(m) || v2.contains(m)
      case VecTimesScalar(v1, _) => v1.contains(m)
      case VecProd(m1, m2) => m1.contains(m) || m2.contains(m)
      case SumMat(m1) => m1.contains(m)
      case MaxMat(m1) => m1.contains(m)
      case _ => false
    }
  }
    
  def contains(v: Vec): Boolean = {
    if (this == v) 
      true 
    else
      this match {
        case VecData(_, _) => false
//        case VecInit(_, _, _) => false
        case x: CudaVec => x.vec.contains(v)
        case x: VecAsIndicator => x.vec.contains(v)
        case VecDec(_, _) => false 
        case PointwiseProd(v1, v2) => v1.contains(v) || v2.contains(v)
        case VecExp(i, e) =>  e.contains(v)
        case VecSlice(v1, i) => v1.contains(v)
        case VecApp(fun, arg) => fun.contains(v) || arg.contains(v)
        case ScalarVecApp(fun, arg) => arg.contains(v) 
        case VecProd(m1, m2) => m1.contains(v) || m2.contains(v)
        case VecFlexSlice(v1, _) => v1.contains(v)
        case FlattenVec(v1, _, _) => v1.contains(v)
        case UnFlattenVec(v1, _, _) => v1.contains(v)
        case VecPlus(v1, v2) => v1.contains(v) || v2.contains(v)  
        case ExpVec(v1) => v1.contains(v)
        case LogVec(v1) => v1.contains(v)
        case VecTimesScalar(v1, e) => v1.contains(v) || e.contains(v)
        case IndicatorVec(v1, v2) => v1.contains(v) || v2.contains(v) 
        case MaxMat(m) => m.contains(v)
        case CopyVec(_, vec) => vec.contains(v)
        case SumMat(m) => m.contains(v)
        case PowVec(v1, _) => v1.contains(v) 
        case VecClip(v1, b) => v1.contains(v)
        case ConcatVec(lst) => lst.exists(l=>l.contains(v))
        case ProjVec(dy, lst, _) => dy.contains(v) || lst.contains(v)
        case FixVec(_, param, dim) => param.exists(p=>p.contains(v)) 
        case FixProd(v1, g) => v1.contains(v) || g.contains(v) 
      } 
  }
  // training parameters
  def freeParam = allParam.filter({case x: VecParam if(x.init.fixed) => false case _ => true }).toList
  // derivable variables (exclude fixed parameters)
  def free: Set[VecDec] = _free.filter({case x: VecParam if(x.init.fixed) => false case _ => true })
  // parameters that need initialization
  def allParam = _free.foldRight[Set[VecParam]](Set())((e, c) => e match {case x:VecParam => c+x case _=> c})
    
  def _free: Set[VecDec] = this match {
//      case VecInit(_, _, _) => Set()
      case VecProd(m1, m2) => m1.vec._free ++ m2.vec._free
      case SumMat(m) => m.vec._free
      case VecData(_, _) => Set()
      case x : VecParam => Set(x)  
      case x : VecAsIndicator => Set()
      case x : CudaVec => Set()
      case x : VecDec => Set(x) 
      case VecSlice(v, _) => v._free
      case VecExp(_, e) => e._free
      case VecApp(fun, arg) => fun.body._free - fun.param ++ arg._free
      case ScalarVecApp(_, arg) => arg._free
      case VecTimesScalar(v, _) => v._free
      case VecFlexSlice(v, _) => v._free
      case PointwiseProd(v1, v2) => v1._free ++ v2._free
      case IndicatorVec(v1, v2) => v1._free ++ v2._free
      case UnFlattenVec(v, _, _) => v._free
      case FlattenVec(v, _, _) => v._free
      case VecPlus(v1, v2) => v1._free ++ v2._free
      case ExpVec(v) => v._free
      case CopyVec(_, v) => v._free
      case LogVec(v) => v._free 
      case PowVec(v, _) => v._free
      case VecClip(v, _) => v._free
      case FixVec(_, param, _) => param.map(p=>p._free).reduce(_++_)  
      case ConcatVec(lst) => lst.map(p=>p._free).reduce(_++_)
      case FixProd(v, FixGrad(_, param, _)) => (v::param).map(p=>p._free).reduce(_++_) 
      case ProjVec(dy, lst, _) => (dy::lst).map(p=>p._free).reduce(_++_)
  }  
  def free(i: IndexDec): Boolean = {
     this match { 
      case VecDec(_,_) => false
      case VecSlice(v, idx) => v.free(i) || idx.exists(index => index.contains(i))
      case VecExp(j, e) => j.forall(k => k!=i) && e.free(i)
      case VecApp(fun, arg) => fun.free(i) || arg.free(i)
      case ScalarVecApp(_, arg) => arg.free(i)
      case VecFlexSlice(v, pattern) => v.free(i) || pattern.exists(p => p.contains(i)) 
      case VecProd(m1, m2) => m1.free(i) || m2.free(i)
      case VecPlus(v1, v2) => v1.free(i) || v2.free(i)
      case VecTimesScalar(v, e) => v.free(i) || e.free(i)
      case ExpVec(v) => v.free(i)
      case LogVec(v) => v.free(i)
      case IndicatorVec(v1, v2) => v1.free(i) || v2.free(i)
      case FlattenVec(v, _, _) => v.free(i)
      case MaxMat(m) => m.free(i)
      case PointwiseProd(v1, v2) => v1.free(i) || v2.free(i)
      case CopyVec(idx, v) => idx.exists(index => index.contains(i)) || v.free(i)
      case UnFlattenVec(v, _, _) => v.free(i) 
      case SumMat(m) => m.free(i) 
      case PowVec(v, _) => v.free(i)
      case VecClip(v, b) => v.free(i)
      case FixVec(_,_,_) => false
      case ConcatVec(_) => false
      case ProjVec(_,_,_) => false
    }
  }
  
  def sub(dl: Dim, dr: Dim) : Vec = { 
    this match {  
      case VecSlice(v, idx) => VecSlice(v.sub(dl, dr), idx.map(dx => dx.sub(dl, dr)))
      case VecExp(idx, e) => VecExp(idx.map(dx => dx.sub(dl, dr)), e.sub(dl, dr))
      case VecApp(fun, arg) => VecApp(fun.sub(dl, dr), arg.sub(dl, dr))
      case ScalarVecApp(fun, arg) => ScalarVecApp(fun, arg.sub(dl, dr))
      case VecTimesScalar(v, e) => VecTimesScalar(v.sub(dl, dr), e.sub(dl, dr))
      case VecProd(m1, m2) => VecProd(m1.sub(dl, dr), m2.sub(dl, dr))
      case VecFlexSlice(v, p) => VecFlexSlice(v.sub(dl, dr), p.map(pt=>pt.sub(dl, dr)))
      case PointwiseProd(v1, v2) => PointwiseProd(v1.sub(dl, dr), v2.sub(dl, dr))
      case IndicatorVec(v1, v2) => IndicatorVec(v1.sub(dl, dr), v2.sub(dl, dr))
      case UnFlattenVec(v, n, d) => UnFlattenVec(v.sub(dl, dr), n, d.map(dim=>dim.sub(dl, dr)))
      case FlattenVec(v, n, d) => FlattenVec(v.sub(dl, dr), n, d.map(dim=>dim.sub(dl, dr)))
      case VecPlus(v1, v2) => VecPlus(v1.sub(dl, dr), v2.sub(dl, dr))
      case ExpVec(v) => ExpVec(v.sub(dl, dr))
      case MaxMat(m) => MaxMat(m.sub(dl, dr))
      case CopyVec(i, v) => CopyVec(i.map(j=>j.sub(dl, dr)), v.sub(dl, dr))
      case LogVec(v) => LogVec(v.sub(dl, dr))
      case SumMat(m) => SumMat(m.sub(dl, dr))
      case PowVec(v, x) => PowVec(v.sub(dl, dr), x.sub(dl, dr)) 
      case VecClip(v, b) => VecClip(v.sub(dl, dr), b)
      case ConcatVec(lst) => ConcatVec(lst.map(l => l.sub(dl, dr))) 
      case ProjVec(dy, lst, n) => ProjVec(dy.sub(dl, dr), lst.map(l => l.sub(dl, dr)), n) 
      case x@FixVec(_,_,_) => x._sub(dl, dr)
      case FixProd(v, g) => FixProd(v.sub(dl, dr), g.sub(dl, dr)) 
    }
  }
  def sub(il: IndexDec, ir: Index): Vec =  
    this match { 
      case x@VecDec(_, _) => x
      case VecSlice(v, idx) => VecSlice(v.sub(il, ir), idx.map(dx => dx.sub(il, ir)))
      case VecExp(idx, e) => VecExp(idx, e.sub(il, ir))
      case VecApp(fun, arg) => VecApp(fun.sub(il, ir), arg.sub(il, ir))
      case ScalarVecApp(fun, arg) => ScalarVecApp(fun, arg.sub(il, ir))
      
      case VecFlexSlice(v, p) => VecFlexSlice(v.sub(il, ir), p.map(pt=>pt.sub(il, ir)))
      case VecProd(m1, m2) => VecProd(m1.sub(il, ir), m2.sub(il, ir))
      case VecPlus(v1, v2) => VecPlus(v1.sub(il, ir), v2.sub(il, ir))
      case VecTimesScalar(v, e) => VecTimesScalar(v.sub(il, ir), e.sub(il, ir))
      case ExpVec(v) => ExpVec(v.sub(il, ir))
      case LogVec(v) => LogVec(v.sub(il, ir))
      case IndicatorVec(v1, v2) => IndicatorVec(v1.sub(il, ir), v2.sub(il, ir))
      case FlattenVec(v, start, flatDims) => FlattenVec(v.sub(il, ir), start, flatDims)
      case UnFlattenVec(v, start, flatDims) => UnFlattenVec(v.sub(il, ir), start, flatDims)
      
      case SumMat(m) => SumMat(m.sub(il, ir)) 
      case VecClip(v, b) => VecClip(v.sub(il, ir), b)
      case PointwiseProd(v1, v2) => PointwiseProd(v1.sub(il, ir), v2.sub(il, ir))
      case PowVec(v1, x) => PowVec(v1.sub(il, ir), x)
    } 

  def sub(vl: Vec, vr: Vec): Vec = 
      if (this == vl) 
        vr 
      else  
        this match { 
//          case x@VecInit(_, _, _) => x
          case x@VecData(_, _) => x
          case x@VecDec(_, _) => x 
          case PointwiseProd(v1, v2) => PointwiseProd(v1.sub(vl, vr), v2.sub(vl, vr))
          case VecExp(i, e) =>  VecExp(i, e.sub(vl, vr))
          case VecSlice(v1, i) => VecSlice(v1.sub(vl, vr), i)
          case VecApp(fun, arg) => VecApp(fun.sub(vl, vr), arg.sub(vl, vr))
          case ScalarVecApp(fun, arg) => ScalarVecApp(fun, arg.sub(vl, vr)) 
          case VecProd(m1, m2) => VecProd(m1.sub(vl, vr), m2.sub(vl, vr))
          case VecFlexSlice(v1, p) => VecFlexSlice(v1.sub(vl, vr), p)
          case FlattenVec(v1, n, d) => FlattenVec(v1.sub(vl, vr), n, d)
          case UnFlattenVec(v1, n, d) => UnFlattenVec(v1.sub(vl, vr), n, d)
          case VecPlus(v1, v2) => VecPlus(v1.sub(vl, vr), v2.sub(vl, vr))
          case ExpVec(v1) => ExpVec(v1.sub(vl, vr))
          case LogVec(v1) => LogVec(v1.sub(vl, vr))
          case VecTimesScalar(v1, e) => VecTimesScalar(v1.sub(vl, vr), e.sub(vl, vr))
          case IndicatorVec(v1, v2) => IndicatorVec(v1.sub(vl, vr), v2.sub(vl, vr))
          case MaxMat(m) => MaxMat(m.sub(vl, vr))
          case CopyVec(i, v) => CopyVec(i, v.sub(vl, vr))
          case SumMat(m) => SumMat(m.sub(vl, vr))
          case PowVec(v, x) => PowVec(v.sub(vl, vr), x) 
          case VecClip(v, b) => VecClip(v.sub(vl, vr), b)
          case ConcatVec(lst) => ConcatVec(lst.map(l=>l.sub(vl, vr)))
          case ProjVec(dy, lst, n) => ProjVec(dy.sub(vl, vr), lst.map(l=>l.sub(vl, vr)), n)
          case FixProd(v, g) =>  FixProd(v.sub(vl, vr), g.sub(vl, vr))
          case x @ FixVec(_,_,_) => x._sub(vl, vr)
        }
  
  // return f(v1) if the vec is in-place computation and return g otherwise 
  def actIfInplace[X](f: Vec=>X, g: X):X = this match {
            case ExpVec(v1)           => f(v1)
            case IndicatorVec(v1, _)  => f(v1) 
            case LogVec(v1)           => f(v1) 
            case VecPlus(v1, _)       => f(v1)
            case PointwiseProd(v1, _) => f(v1)  
            case PowVec(v1, _)        => f(v1) 
            case ScalarVecApp(f1, v1) => f(v1)
            case VecTimesScalar(v1, _)=> f(v1)
            case VecClip(v1, _)       => f(v1)
            case FixVec(a, List(v1), _) if a.isInstanceOf[Activation] 
              => f(v1)
            case FixProd(v1, grad) if grad.v.layer.isInstanceOf[Activation] || 
                                             grad.v.layer.isInstanceOf[LRN] ||
                                             grad.v.layer.isInstanceOf[BatchNorm] && grad.x == grad.v.param(0)
              => f(v1)
            case _ => g
    }
  // return v with v1 replaced by f(v1) if in-place computation and return "this" otherwise
  def subIfInplace(f: Vec=>Vec) = this match {
            case ExpVec(v1)           => ExpVec(f(v1))
            case IndicatorVec(v1, v2) => IndicatorVec(f(v1), v2) 
            case LogVec(v1)           => LogVec(f(v1)) 
            case VecPlus(v1, v2)      => VecPlus(f(v1), v2)
            case PointwiseProd(v1, v2)=> PointwiseProd(f(v1), v2)  
            case PowVec(v1, v2)       => PowVec(f(v1), v2) 
            case ScalarVecApp(f1, v1) => ScalarVecApp(f1, f(v1))
            case VecTimesScalar(v1,e) => VecTimesScalar(f(v1), e)
            case VecClip(v1, b)       => VecClip(f(v1), b)
            case FixVec(a, List(v1), dim) if a.isInstanceOf[Activation] 
              => FixVec(a, List(f(v1)), dim)
            case FixProd(v1, grad) if grad.v.layer.isInstanceOf[Activation] || grad.v.layer.isInstanceOf[LRN] 
              => FixProd(f(v1), grad)
            case x:Vec => x
  }
}
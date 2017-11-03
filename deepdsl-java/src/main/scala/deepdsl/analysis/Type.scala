package deepdsl.analysis

import deepdsl.derivation._
import deepdsl.run.Env
import deepdsl.layer._
import deepdsl.ast.VecTimesScalar
import deepdsl.ast.VecSlice
import deepdsl.ast.VecPlus
import deepdsl.ast.VecExp
import deepdsl.ast.VecDec
import deepdsl.ast.VecClip
import deepdsl.ast.VecApp
import deepdsl.ast.Vec
import deepdsl.ast.ScalarVecApp
import deepdsl.ast.FixVec
import deepdsl.ast.ConcatVec
import deepdsl.ast.DimVar
import deepdsl.ast.Dim
import deepdsl.ast.VecFun

case class TypeException(e: String) extends RuntimeException(e)

object typeof {
	type Unifier = List[(DimVar, Dim)]
	type VecType = List[Dim]
	
  def apply(f: VecFun): (VecType, VecType) = {
    (f.param.dim, apply(f.body, List()))
  }
  def apply(v: Vec, unifier: Unifier): VecType = v match {
      case VecApp(fun, arg) => typeof(fun.body, unify(typeof(fun.param, unifier), typeof(arg, unifier)))
     
      case VecDec(v, dim) => substitute(dim, unifier)
      case VecSlice(v, idx) => apply(v, unifier)
      case VecExp(i, e1) => substitute(v.getDims, unifier)
 
      case ScalarVecApp(fun, arg) => apply(arg, unifier)
      case VecTimesScalar(v, e) => apply(v, unifier)
//      case VecProd(m1, m2) =>  
//      case VecFlexSlice(v, p) =>  
//      case PointwiseProd(v1, v2) =>  
//      case IndicatorVec(v1, v2) =>  
//      case UnFlattenVec(v, n, d) =>  
//      case FlattenVec(v, n, d) =>   
//      case ExpVec(v) =>  
//      case MaxMat(m) => 
//      case CopyVec(i, v) =>  
//      case LogVec(v) => 
//      case SumMat(m) =>  
//      case PowVec(v, x) =>  
      case VecClip(v, b) => apply(v, unifier)
      case FixVec(name, param, dim) => substitute(dim, unifier)
      case ConcatVec(lst) => substitute(v.getDims, unifier)
      case VecPlus(v1, v2) => {
        val t1 = apply(v1, unifier)
        val t2 = apply(v2, unifier)
//        if(t1 != t2) {
//          throw new RuntimeException(s"type $t1 doesn't match $t2") 
//        }
        t1
      }
      case _ => throw new TypeException(s"unsupported IR expression $v for type checking")
  }
 
  def substitute(typ: VecType, unifier: Unifier) = {
    typ.map(t => unifier.foldLeft(t)((c, e) => c.sub(e._1, e._2)))
  }
//  def substitute(v: Vec, unifier: Unifier) = {
//    unifier.foldLeft(v)((c, e) => c.sub(e._1, e._2))
//  }
//  def substitute(x: AExp, unifier: Unifier) = {
//    unifier.foldLeft(x)((c, e) => c.sub(e._1, e._2))
//  }
  def substitute(d: Dim, unifier: Unifier) = {
    unifier.foldLeft(d)((c, e) => c.sub(e._1, e._2))
  }
  def unify(lt: VecType, rt: VecType): Unifier = {
    if(lt.size != rt.size) {
       throw new TypeException(lt + "\t!=\t" + rt)  
    }
    else { 
       (lt zip rt).foldLeft[Unifier](Nil)((c, e) => e match {
          case (x @ DimVar(_), y: Dim) => (x, y) :: c
          case _ => {
            if(! e._1.toString.equals(e._2.toString)) {
              throw new TypeException(e._1 + "\t!=\t" + e._2)  
            }
            c
          }
       })
    }
  }
}

//trait Type { }
//
//case class NumType() extends Type
//
//case class IndexType(d: Dim) extends Type
//
//case class VecType(d: List[Dim]) extends Type
//
//case class MatType(d1: List[Dim], d2: List[Dim]) extends Type
//
//case class FunType(param: Type, result: Type) extends Type
//
//case class TypEnv(it: Map[IndexDec, IndexType], vt: Map[VecDec, VecType], dt: Map[DimVar, Range]) extends Env(dt) {
//  def apply(i: IndexDec) = it(i)
//  def apply(v: VecDec) = vt(v) 
//}
//
//object infer {
//  def apply(d: Dim, env: Env): DimConst = DimConst(env(d).size)
//  def apply(i: IndexDec, env: Env): IndexDec = IndexDec(i.idx, infer(i.dim, env))
//  
//  def apply(e: AExp, env: Env): AExp =  e match { 
//      case Plus(e1, e2) => Plus(infer(e1, env), infer(e2, env)) 
//      case Times(e1, e2) => Times(infer(e1, env), infer(e2, env))
//      case Sum(i, e1) => { val i1 = infer(i, env); Sum(i1, e1.sub(i, i1)) }
//      case VecElem(vec, idx) => VecElem(infer(vec, env).asInstanceOf[VecDec], idx)
//      	
//      case Delta(il, ir, e1) => Delta(il, ir, infer(e1, env))
//      case Log(e) => Log(infer(e, env))
//      case Exp(e) => Exp(infer(e, env))
//      case Pow(e, x) => Pow(infer(e, env), x)
//      case Max(i, e1) => { 
//        val i1 = i.map(j=>infer(j, env))
//        Max(i1, sub(infer(e1, env), i, i1)) 
//      }
//      case ScalarApp(fun, arg) => ScalarApp(fun, infer(arg, env))
//      case ScalarDec(name) => e
//      case Vec2ScalarApp(fun, arg) => {
//        val arg1 = infer(arg, env)
//        val d1 = arg1.getDims.map(d => infer(d, env))
//        val v1 = VecDec(fun.param.v, d1)
//        Vec2ScalarApp(Vec2ScalarFun(v1, infer(fun.body.redex(fun.param, v1), env)), arg1)
//      }
//      case InnerProd(v1, v2) => InnerProd(infer(v1, env), infer(v2, env)) 
//      case Num(_) | Zero => e
//      case DimSize(d) => DimSize(infer(d, env))
//  }
////  def apply(v: VecDec, env: Env): VecDec = VecDec(v.v, v.dim.map(d => infer(d, env)))
//  def apply(v: Vec, env: Env): Vec = v match {
//      case v: VecData => v
//      case v: VecConst => v
//      case v: VecDecCopy => new VecDecCopy(infer(v.dec, env))
//      case v: VecAsIndicator => v
//      case VecDec(v, dim) => VecDec(v, dim.map(d => infer(d, env)))
//      case VecSlice(v, idx) => VecSlice(infer(v, env), idx)
//      case VecExp(i, e1) => {
//        val i1 = i.map(j=>infer(j, env))
//        VecExp(i1, sub(infer(e1, env), i, i1))
//      }
//      case VecApp(fun, arg) => {
//        val arg1 = infer(arg, env)
//      	val d1 = arg1.getDims.map(d => infer(d, env))
//      	val v1 = VecDec(fun.param.v, d1)
//      	VecApp(VecFun(v1, infer(fun.body.redex(fun.param, v1), env)), arg1)
//      }
//      case ScalarVecApp(fun, arg) => ScalarVecApp(fun, infer(arg, env))
//      case VecTimesScalar(v, e) => VecTimesScalar(infer(v, env), infer(e, env))
//      case VecProd(m1, m2) => VecProd(infer(m1, env), infer(m2, env))
//      case VecFlexSlice(v, p) => VecFlexSlice(infer(v, env), p)
//      case PointwiseProd(v1, v2) => PointwiseProd(infer(v1, env), infer(v2, env))
//      case IndicatorVec(v1, v2) => IndicatorVec(infer(v1, env), infer(v2, env))
//      case UnFlattenVec(v, n, d) => UnFlattenVec(infer(v, env), n, d)
//      case FlattenVec(v, n, d) => FlattenVec(infer(v, env), n, d)
//      case PlusVec(v1, v2) => PlusVec(infer(v1, env), infer(v2, env))
//      case ExpVec(v) => ExpVec(infer(v, env))
//      case MaxMat(m) => MaxMat(infer(m, env))
//      case CopyVec(i, v) => CopyVec(i.map(j=> infer(j, env)), infer(v, env))
//      case LogVec(v) => LogVec(infer(v, env))
//      case SumMat(m) => SumMat(infer(m, env))
//      case PowVec(v, x) => PowVec(infer(v, env), x) 
//      case VecClip(v, b) => VecClip(infer(v, env), b)
//  }
//  
//  def apply(m: Mat, env: Env) : Mat = m match {
//      case MatDec(name, i, v) => {
//        val i1 = i.map(j => infer(j, env))
//        MatDec(name, i1, sub(infer(v, env), i, i1))
//      }
//      case MatExp(i, v) => {
//        val i1 = i.map(j => infer(j, env))
//        MatExp(i1, sub(infer(v, env), i, i1))
//      }
//      case MatOfVec(i, v) => {
//        val i1 = i.map(j => infer(j, env))
//        MatOfVec(i1, infer(v, env))
//      }
//      case MatOfVecFlexSlice(i, v) => {
//        val i1 = i.map(j => infer(j, env))
//        MatOfVecFlexSlice(i1, VecFlexSlice(sub(infer(v.v, env), i, i1), v.p.map(p1 => sub(p1, i, i1))))
//      }
//  }
//  
//  def sub(e1: AExp, il: Seq[IndexDec], ir: Seq[IndexDec]) = {
//      (il zip ir).foldLeft(e1)((c, e) => c.sub(e._1, e._2))
//  }
//  def sub(v: Vec, il: Seq[IndexDec], ir: Seq[IndexDec]) = {
//      (il zip ir).foldLeft(v)((c, e) => c.sub(e._1, e._2))
//  }
//  def sub(p: Pattern, il: Seq[IndexDec], ir: Seq[IndexDec]) = {
//      (il zip ir).foldLeft(p)((c, e) => c.sub(e._1, e._2))
//  }
//}
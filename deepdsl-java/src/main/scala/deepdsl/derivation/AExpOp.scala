package deepdsl.derivation

import deepdsl.ast._

trait AExpOp { 
  override def toString = this match { 
        case IndexExp(i) => i.toString 
        case Indicator(e, indexedExp) => "1{" + e + " == " + indexedExp + "}" 
      	case Plus(e1, Times(e2, Num(-1))) => "(" + e1 + " - " + e2 + ")" 
      	case Plus(Times(e1, Num(-1)), e2) => "(" + e2 + " - " + e1 + ")"
      	case Plus(e1, e2) => "(" + e1 + " + " + e2 + ")"
      	case Times(e1, Pow(e2, Num(-1))) => "(" + e1 + " / " + e2 + ")"
      	case Times(Pow(e1, Num(-1)), e2) => "(" + e2 + " / " + e1 + ")"
      	case Times(e1, e2) => "(" + e1 + " * " + e2 + ")"
      	case Sum(i, e1) => "sum(" + i + ", " + e1 + ")"
      	case VecElem(vec, idx) => vec + "[" + idx.mkString(", ") + "]"
      	case Delta(il, ir, e1) => "([" + il + "\\" + ir + "] " + e1 + ")"
      	case Log(e) => "log " + e
      	case Exp(e) => "e^" + e
      	case Pow(e, x) => e.toString + "^" + x 
      	case E => "e"   
      	case Max(i, v) => "max([" + i.mkString(", ") + "], " + v + ")"
      	case ScalarApp(fun, arg) =>  "(" + fun + " " + arg +")"
      	case ScalarDec(name) => name
      	case Vec2ScalarApp(fun, arg) => "(" + fun + " " + arg + ")" 
      	case Real(x, name) => ""+x
      	case InnerProd(v1, v2) => "(" + v1 + " . " + v2 + ")"  
      	case MaxVec(v) => "Max(" + v + ")" 
      	case SumVec(v) => "Sum(" + v + ")"
      	case Accuracy(p, y, k) => s"Accuracy($p, $y, $k)"
  } 
  
  // training parameters
  def freeParam = allParam.filter({case x: VecParam if(x.init.fixed) => false case _ => true }).toList
  // parameters that need initialization
  def allParam = _free.foldRight[Set[VecParam]](Set())((e, c) => e match {case x:VecParam => c+x case _=> c})
  // derivable variables (exclude fixed parameters)
  def free: Set[VecDec] = _free.filter({case x: VecParam if(x.init.fixed) => false case _ => true })
  
  def _free: Set[VecDec] = this match {
      case Plus(e1, e2) 	=> e1._free ++ e2._free
    	case Times(e1, e2) 	=> e1._free ++ e2._free
    	case VecElem(v, _)=> v._free
    	case Sum(_, e1) 		=> e1._free
    	case Exp(e1) 			=> e1._free
    	case Pow(e1, _) 		=> e1._free
    	case Log(e1) 			=> e1._free
    	case Delta(_, _, e1) 	=> e1._free
    	case Max(_, x) 		=> x._free
    	case ScalarApp(_, a)	=> a._free
    	case Vec2ScalarApp(f, a) => f.body._free - f.param ++ a._free
    	case Indicator(e1, e2) 	=> e1._free ++ e2._free
    	case SumVec(v) => v._free
    	case IndexExp(j) => Set()
    	case InnerProd(v1, v2) => v1._free ++ v2._free
    	case MaxVec(v) => v._free
    	case Num(_) | DimSize(_) | Real(_, _) => Set() 
    	case Accuracy(p, y, _) => p._free
  }
  def sub(dl: Dim, dr: Dim): AExp = this match {
    	  case Plus(e1, e2) 	=> Plus(e1.sub(dl, dr), e2.sub(dl, dr))
    	  case Times(e1, e2) 	=> Times(e1.sub(dl, dr), e2.sub(dl, dr))
    	  case VecElem(vec, idx)=> VecElem(vec.sub(dl, dr), idx.map(index => index.sub(dl, dr)))
    	  case Sum(i, e1) 		=> Sum(i.sub(dl, dr), e1.sub(dl, dr))
    	  case Exp(e1) 			=> Exp(e1.sub(dl, dr))
    	  case Pow(e1, x) 		=> Pow(e1.sub(dl, dr), x)
    	  case Log(e1) 			=> Log(e1.sub(dl, dr)) 
    	  case Delta(i, j, e1) 	=> Delta(i.sub(dl, dr), j.sub(dl, dr), e1.sub(dl, dr))  
    	  case Max(i, x) 		=> Max(i.map(idx=>idx.sub(dl, dr)), x.sub(dl, dr))		
    	  case ScalarApp(f, a)	=> ScalarApp(f, a.sub(dl, dr)) 
    	  case Vec2ScalarApp(f, a) => Vec2ScalarApp(f.sub(dl, dr), a.sub(dl, dr))
    	  case Indicator(e1, e2) 	=> Indicator(e1.sub(dl, dr), e2.sub(dl, dr))
    	  case SumVec(v) => SumVec(v.sub(dl, dr))
    	  case IndexExp(j) => IndexExp(j.sub(dl, dr))
    	  case InnerProd(v1, v2) => InnerProd(v1.sub(dl, dr), v2.sub(dl, dr)) 
    	  case MaxVec(v) => MaxVec(v.sub(dl, dr))
    	  case x@Real(_, _) => x
    	  case Accuracy(p, y, k) => Accuracy(p.sub(dl, dr), y.sub(dl, dr), k)
  }
  
  def sub(il: IndexDec, ir: Index) : AExp = this match { 
    	  case Plus(e1, e2) 	=> Plus(e1.sub(il, ir), e2.sub(il, ir))
    	  case Times(e1, e2) 	=> Times(e1.sub(il, ir), e2.sub(il, ir))
    	  case VecElem(vec, idx)=> VecElem(vec, idx.map(index => index.sub(il, ir)))
    	  case x@Sum(i, e1) 		=> if(i==il) x else Sum(i, e1.sub(il, ir))
    	  case Exp(e1) 			=> Exp(e1.sub(il, ir))
    	  case Pow(e1, x) 		=> Pow(e1.sub(il, ir), x)
    	  case Log(e1) 			=> Log(e1.sub(il, ir)) 
    	  case Delta(i, j, e1) 	=> Delta(i.sub(il, ir), j.sub(il, ir), e1.sub(il, ir)) // TODO: ugly
    	  case x@Max(i, e1) 		=> if(i.contains(il)) x else Max(i, e1.sub(il, ir))		
    	  case ScalarApp(f, a)	=> ScalarApp(f, a.sub(il, ir)) 
    	  case Vec2ScalarApp(f, a) => Vec2ScalarApp(f.sub(il, ir), a.sub(il, ir)) // TODO: what about argument "a"
    	  case MaxVec(v) => MaxVec(v.sub(il, ir))
    	  case InnerProd(v1, v2) => InnerProd(v1.sub(il, ir), v2.sub(il, ir))
    	  case Indicator(e1, e2) 	=> Indicator(e1.sub(il, ir), e2.sub(il, ir))
    	  case IndexExp(j) 		=> IndexExp(j.sub(il, ir))
    	  case SumVec(v) => SumVec(v.sub(il, ir))  
    	  case x @ Real(_, _) => x
    	  case x @ Accuracy(_, _, _) => x
  }
  
  def sub(vl: ScalarDec, vr: AExp) : AExp = this match {
    	  case Plus(e1, e2) 	=> Plus(e1.sub(vl, vr), e2.sub(vl, vr))
    	  case Times(e1, e2) 	=> Times(e1.sub(vl, vr), e2.sub(vl, vr)) 
    	  case Sum(i, e1) 		=> Sum(i, e1.sub(vl, vr))
    	  case Exp(e1) 			=> Exp(e1.sub(vl, vr))
        case Pow(e1, x) 		=> Pow(e1.sub(vl, vr), x)
    	  case Log(e1) 			=> Log(e1.sub(vl, vr))
    	  case Indicator(e1, e2)=> Indicator(e1.sub(vl, vr), e2.sub(vl, vr))
    	  case Delta(i, j, e1) 	=> Delta(i, j, e1.sub(vl, vr))
    	  case Max(i, x) 		=> Max(i, x.sub(vl, vr))
    	  case ScalarApp(f, a) 	=> ScalarApp(f.sub(vl, vr), a.sub(vl, vr))
    	  case x @ ScalarDec(_) => if (x == vl) vr else x
    	  case x : AExp => x
  }

  def free(i: IndexDec): Boolean = this match {
        case Times(e1, e2) => e1.free(i) || e2.free(i)
        case Plus(e1, e2) => e1.free(i) || e2.free(i)
        case Sum(j, e2) => i != j && e2.free(i)
        case Pow(e1, _) => e1.free(i)
        case Exp(e1) => e1.free(i)
        case Log(e1) => e1.free(i)
        case Delta(il, ir, e) => il==i || ir==i || e.free(i) 
        case VecElem(_, idx) => idx.exists(index => index.contains(i)) 
        case ScalarApp(_, arg) => arg.free(i) // we do not model closure with our symbolic function
        case Max(idx, e1) =>   e1.free(i) 
        case Vec2ScalarApp(fun, arg) => fun.free(i) || arg.free(i)
        case ScalarDec(_) => false 
        case InnerProd(v1, v2) => v1.free(i) || v2.free(i) 
        case Indicator(e1, e2) => e1.free(i) || e2.free(i)
        case IndexExp(j) => j.contains(i)
        case SumVec(v) => v.free(i) 
        case MaxVec(v) => v.free(i)
        case Real(_, _) => false
        case Accuracy(_, _, _) => false
  }
  
  def contains(v: Vec): Boolean = 
    this match {
        case VecElem(vec, _) => vec == v 
        case Num(_) | IndexExp(_) | DimSize(_) | Real(_, _) => false
        case InnerProd(v1, v2) => v1.contains(v) || v2.contains(v)
        case Plus(e1, e2) => e1.contains(v) || e2.contains(v)
        case Times(e1, e2) =>  e1.contains(v) || e2.contains(v)   
        case Delta(il, ir, e1) => e1.contains(v)
        case Log(e1) => e1.contains(v)
        case Exp(e1) => e1.contains(v)
        case Pow(e1, x) => e1.contains(v)
        case Max(i, v1) => v1.contains(v)
        case ScalarApp(fun, arg) =>  arg.contains(v) 
        case Vec2ScalarApp(fun, arg) =>  fun.contains(v) || arg.contains(v)
        case Sum(i, e) => e.contains(v)
        case MaxVec(v1) => v1.contains(v)
        case Indicator(e1, e2) => e1.contains(v) || e2.contains(v)
        case SumVec(v1) => v1.contains(v)
        case Accuracy(p, y, _) => p.contains(v) || y.contains(v)
    } 
  
  def sub(vl: Vec, vr: Vec): AExp = 
    this match {
        case x @ VecElem(v, i) => if (v == vl) vr.get(i) else x 
        case x @ Num(_) => x
        case x @ IndexExp(_) => x 
        case x @ DimSize(_) => x
        case x @ Real(_, _) => x
        case InnerProd(v1, v2) => InnerProd(v1.sub(vl, vr), v2.sub(vl, vr))
        case Plus(e1, e2) => Plus(e1.sub(vl, vr), e2.sub(vl, vr))
        case Times(e1, e2) =>  Times(e1.sub(vl, vr), e2.sub(vl, vr))   
        case Delta(il, ir, e1) => Delta(il, ir, e1.sub(vl, vr))
        case Log(e1) => Log(e1.sub(vl, vr))
        case Exp(e1) => Exp(e1.sub(vl, vr))
        case Pow(e1, x) => Pow(e1.sub(vl, vr), x)
        case Max(i, e1) => Max(i, e1.sub(vl, vr))
        case ScalarApp(fun, arg) =>  ScalarApp(fun, arg.sub(vl, vr))
        case Vec2ScalarApp(fun, arg) => Vec2ScalarApp(fun.sub(vl, vr), arg.sub(vl, vr))
        case Sum(i, e1) => Sum(i, e1.sub(vl, vr))
        case MaxVec(v1) => MaxVec(v1.sub(vl, vr))
        case Indicator(e1, e2) => Indicator(e1.sub(vl, vr), e2.sub(vl, vr))
        case SumVec(v) => SumVec(v.sub(vl, vr)) 
        case Accuracy(p, y, top) => Accuracy(p.sub(vl, vr), y, top)
    } 
}
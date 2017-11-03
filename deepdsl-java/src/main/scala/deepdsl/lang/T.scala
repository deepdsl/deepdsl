package deepdsl.lang

import deepdsl.derivation._
import scala.language.implicitConversions 
import deepdsl.ast._
import deepdsl.ast.Init
import deepdsl.ast.Data

object T {
  implicit def index2Exp(i: Index) = IndexExp(i)
  implicit def int2Exp(n: Int) = Num(n)

  def let(v: Vec, f: VecDec => Vec) = {
	  val x = T._new(v.getDims)
    VecFun(x, f(x))(v)
  }
  def fun(f: ScalarDec => AExp) = {
    val x = ScalarDec("x")
    ScalarFun(x, f(x))
  } 
  def fun(d1: Dim, d2: Dim, f: VecDec => AExp) = {
    fun_app(List(d1, d2), (x:VecDec) => f(x))
  }
  def fun(d1: Dim, f: VecDec => AExp) = {
    fun_app(List(d1), (x:VecDec) => f(x))
  }
  private def fun_app(d: List[Dim], f: VecDec => AExp) = {
    val x = T._new(d)
    Vec2ScalarFun(x, f(x))
  } 
  def fun(n: Int, f: VecDec => Vec) = {
    fun_app((1 to n).map(_=>T.dim).toList, f)
  }
  def fun(d1: Dim, d2: Dim, d3: Dim, d4: Dim, f: VecDec => Vec) = { 
    fun_app(List(d1, d2, d3, d4), f) 
  }
  def fun(d1: Dim, d2: Dim, d3: Dim, f: VecDec => Vec) = { 
    fun_app(List(d1, d2, d3), f) 
  }
  def fun(d1: Dim, d2: Dim, f: VecDec => Vec) = { 
    fun_app(List(d1, d2), f) 
  }
  def fun(d1: Dim, f: VecDec => Vec) = { 
    fun_app(List(d1), f) 
  }
  private def fun_app(d: List[Dim], f: VecDec => Vec) = {
    val x = T._new(d)
    VecFun(x, f(x))
  } 

	def sum(d1: Dim, d2: Dim, d3: Dim, f: (IndexDec, IndexDec, IndexDec) => AExp) : AExp = {
    sum_app(List(d1, d2, d3), (i: List[IndexDec]) => f(i(0), i(1), i(2)))
  }
	def sum(d1: Dim, d2: Dim, f: (IndexDec, IndexDec) => AExp) : AExp = {
    sum_app(List(d1, d2), (i: List[IndexDec]) => f(i(0), i(1)))
  }
  def sum(d: Dim, f: IndexDec => AExp) : AExp = {
    sum_app(List(d), (i: List[IndexDec]) => f(i(0)))
  }
  private def sum_app(d: List[Dim], f: List[IndexDec] => AExp) = {
    val i = d.map(dim=>T.index(dim))
    i.foldRight(f(i))((e, c) => Sum(e, c))
  } 
  def max(d1: Dim, d2: Dim, f: (IndexDec, IndexDec) => AExp): AExp = {
    max_app(List(d1, d2), (i:List[IndexDec]) => f(i(0), i(1)))
  }
  def max(d: Dim, f: IndexDec => AExp) : AExp = {
    max_app(List(d), (i:List[IndexDec]) => f(i(0)))
  }
  private def max_app(d: List[Dim], f: List[IndexDec] => AExp) = {
    val i = d.map(dim=>T.index(dim))
    Max(i, f(i))
  }
  def vec(d1: Dim, d2: Dim, d3: Dim, d4: Dim, f: (IndexDec, IndexDec, IndexDec, IndexDec) => AExp) = {
    vec_app(List(d1,d2,d3,d4), (i:List[IndexDec]) => f(i(0),i(1),i(2),i(3)))
  }
  def vec(d1: Dim, d2: Dim, d3: Dim, f: (IndexDec, IndexDec, IndexDec) => AExp) = {
    vec_app(List(d1,d2,d3), (i:List[IndexDec]) => f(i(0),i(1),i(2)))
  }
  def vec(d1: Dim, d2: Dim, f: (IndexDec, IndexDec) => AExp) = {
    vec_app(List(d1,d2), (i:List[IndexDec]) =>f(i(0), i(1)))
  }
  def vec(d: Dim, f: IndexDec => AExp) = {
    vec_app(List(d), (i:List[IndexDec]) => f(i(0)))
  }
  private def vec_app(d:List[Dim], f: List[IndexDec] => AExp) = {
    val i = d.map(dim=>T.index(dim))
    VecExp(i, f(i))
  }  
  
  var vec_count = 0  
  def _new(init: Init, name: String, dim: Dim*): VecParam = new VecParam(init, VecVar(name), dim.toList)
  def _new(name: String, dim: Dim*): VecDec = VecDec(VecVar(name), dim.toList)
  def _new(dim: Dim*) : VecDec = _new(dim.toList)
  def _new(dim: List[Dim]): VecDec = { vec_count = vec_count + 1; VecDec(VecVar("X"+ vec_count), dim) }
  def _new(n: Int): VecDec = { _new(1.to(n).map(i => T.dim).toList) } 
  def _new(name: String, dim: List[Int]) = VecDec(VecVar(name), dim.map(d=>T.dim(d)))
 
  var index_count = 0
  private def index = {index_count = index_count + 1; IndexVar("i"+ index_count)}
  def index(dim: Dim): IndexDec = IndexDec(index, dim)
  def index(name: String, dim: Dim) = IndexDec(IndexVar(name), dim)
  val dummy_index = IndexDec(IndexVar("@"), DimConst(1))
  
  var dim_count=0
  def dim = {dim_count = dim_count+1; DimVar("N" + dim_count)}
  def dim(n:Int) = DimConst(n)
}


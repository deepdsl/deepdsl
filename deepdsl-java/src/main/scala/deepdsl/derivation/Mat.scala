package deepdsl.derivation

import deepdsl.lang._
import deepdsl.ast._

//trait IR  

// 2-D tensor (or "matrix")
trait Mat {
  def i : List[IndexDec]
  def vec: Vec

  def free(i: IndexDec) : Boolean = vec.free(i)
  def contains(v: Vec) : Boolean = vec.contains(v)
  
  def contains(m: MatDec): Boolean = this match {
    case MatDec(_,_,_) => this == m
    case MatExp(_, v) => v.contains(m)
    case MatOfVec(_, v) => v.contains(m)
    case MatOfVecFlexSlice(_, v) => v.contains(m)
  }
  
  def sub(vl: Vec, vr: Vec): Mat = this match {
    case MatDec(_,_,_) => this
    case MatExp(i, v) =>  MatExp(i, v.sub(vl, vr)) 
    case MatOfVec(i, v) => MatOfVec(i, v.sub(vl, vr))
    case MatOfVecFlexSlice(i, v) => MatOfVecFlexSlice(i, v.sub(vl, vr).asInstanceOf[VecFlexSlice])
  }
  def sub(il: IndexDec, ir: Index): Mat = this match {
    case MatExp(i, vec) => MatExp(i, vec.sub(il, ir))
    case MatOfVec(i, v) => MatOfVec(i, v.sub(il, ir))
    case MatOfVecFlexSlice(i, vec) => MatOfVecFlexSlice(i, vec.sub(il, ir).asInstanceOf[VecFlexSlice])
  }
  def sub(dl: Dim, dr: Dim): Mat = this match {
    case MatExp(i, vec) => MatExp(i.map(j=>j.sub(dl, dr)), vec.sub(dl, dr))
    case MatOfVec(i, v) => MatOfVec(i.map(j=>j.sub(dl, dr)), v.sub(dl, dr))
    case MatOfVecFlexSlice(i, vec) => MatOfVecFlexSlice(i.map(j=>j.sub(dl, dr)), vec.sub(dl, dr).asInstanceOf[VecFlexSlice])
  }
  
  override def equals(that: Any) = {
    if (that.isInstanceOf[Mat]) {
      val thatMat = that.asInstanceOf[Mat]
      val vec2 = (thatMat.i zip i).foldLeft(thatMat.vec)((c, e) => c.sub(e._1, e._2))
      vec == vec2 
    }
    else
      false
  }
}

object Mat {
  var count = 0;
  def _new(m: Mat) = { count = count + 1; MatDec("m"+count, m.i, m.vec) }
}
case class MatDec(name: String, i: List[IndexDec], vec: Vec) extends Mat { 
  override def toString = name 
  override def contains(v: Vec) = false
  override def equals(that: Any) = {
    if(that.isInstanceOf[MatDec]) {
      val thatDec = that.asInstanceOf[MatDec]
      name.equals(thatDec.name)
    }
    else 
      false
  }
} 

case class MatExp(i: List[IndexDec], vec: Vec) extends Mat { 
  override def toString = "(" + i.mkString(", ") + ") => " + vec  
}
// vec = VecSlice(source_vec, i) 
case class MatOfVec(i: List[IndexDec], source_vec: Vec) extends Mat {
  val vec = source_vec.of(i:_*)
  
  override def toString = "(" + vec.v + ")"+ "(" + i.mkString(", ") + " | " + (1 to vec.size).map(_=>T.dummy_index).mkString(", ") +  ")"
}

case class MatOfVecFlexSlice(i: List[IndexDec], vec: VecFlexSlice) extends Mat {
  override def toString = "(" + i.mkString(", ") + ") => " + vec    
}

trait Pattern { 
  def contains(i: IndexDec) : Boolean = this match { 
    case SlicePattern(j) => j.contains(i)
    case ShiftPattern(j, _) => j.contains(i)
    case ShiftFlipPattern(j, _) => j.contains(i)  
    case _ => false
  }
  def sub(il: IndexDec, ir: Index): Pattern = this match {
    case IndexPattern(_) => this
    case SlicePattern(j) => SlicePattern(j.sub(il, ir))
    case ShiftPattern(j, i) => ShiftPattern(j.sub(il, ir), i) 
    case ShiftFlipPattern(j, i) => ShiftFlipPattern(j.sub(il, ir), i)  
    case UpSamplePattern(c, i) => UpSamplePattern(c.sub(il, ir), i) 
  }
  def sub(dl: Dim, dr: Dim): Pattern = this match {
    case IndexPattern(j) => IndexPattern(j.sub(dl, dr))
    case SlicePattern(j) => SlicePattern(j.sub(dl, dr))
    case ShiftPattern(j, i) => ShiftPattern(j.sub(dl, dr), i.sub(dl, dr)) 
    case ShiftFlipPattern(j, i) => ShiftFlipPattern(j.sub(dl, dr), i.sub(dl, dr))  
    case UpSamplePattern(c, i) => UpSamplePattern(c.sub(dl, dr), i.sub(dl, dr)) 
  }
}
case class IndexPattern(i: IndexDec) extends Pattern 
case class SlicePattern(j: Index) extends Pattern  
case class ShiftPattern(j: Index, i: IndexDec) extends Pattern  
case class ShiftFlipPattern(j: Index, i: IndexDec) extends Pattern 
case class UpSamplePattern(c: Const, i: IndexDec) extends Pattern  


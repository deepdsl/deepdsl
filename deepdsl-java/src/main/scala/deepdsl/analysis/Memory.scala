package deepdsl.analysis

import deepdsl.derivation._
import deepdsl.optimization._
import deepdsl.run.Env
import deepdsl.ast._

/* 
 * Scheduling algorithm: Reorder the statements to reduce max memory 
 *   1. Let statements form a directed acyclic graph (DAG) based on def_dependency -- s1 -> s2 iff s2 depends on s1. 
 *      We say s1 is the parent of s2 and s2 is the child of s1 in this DAG.
 *   2. Height of a node is initialized to 0 if it has no parent.
 *   3. For other node, initialize its height as the maximum height of its parents + 1
 *   // 4. Update the height of each "s" with children to the minimum height of its children - 1.
 *   4. Find a list of terminal node such as parameter gradients that have no children
 *   5. In increasing order of heights, for each terminal node d_w, schedule the ancestor nodes of d_w 
 *   that has not been scheduled in increasing order of their heights.
*/

object schedule {
  def apply(lst: List[Let]) = {
    val (parents, children) = def_dependency(lst)
     
    val heights = scala.collection.mutable.Map[Let, Int]() // initial heights
    def init_height(l : Let) : Int = {
      if(heights.contains(l)) heights(l) else {
        val r = if(parents(l).size == 0) 0 else parents(l).map(p => init_height(p)).max + 1
        heights += l -> r
        r
      }
    }
    lst.foreach(l => init_height(l))
//    val ranks = scala.collection.mutable.Map[Let, Int]() // updated heights
//    def update_height(l : Let) : Int = {
//      if(ranks.contains(l)) ranks(l) else {
//        val r = if(children(l).size == 0) heights(l) else children(l).map(c => update_height(c)).min - 1
//        ranks += l -> r
//        r
//      }
//    }
//    lst.foreach(l => update_height(l))
    
    // find set of ancestors of let (including let) that have not been scheduled
    def ancestors(let: Let, scheduled: List[Let]): Set[Let] = {
      if(scheduled.contains(let)) Set()
      else {
        parents(let).foldLeft(Set(let))((c, e) => c ++ ancestors(e, scheduled++c))  
      }
    }
    
    // find a list of let without children (terminal statements -- i.e. parameter gradients and loss)
    val terminals = lst.filter(l => children(l).size == 0).sortBy(x => heights(x))
    
    // schedule terminals and their ancestors
    terminals.foldLeft(List[Let]())((c, e) => c ++ ancestors(e, c).toList.sortBy(x => heights(x)))
    
//    ranks.toList.sortBy((x) => x._2).unzip._1
  }
}

// map of "VecLet" and the list of lets that use the "VecLet"
object use_dependency {
  def apply(lst: List[Let]) = {
    lst.foldRight[Map[VecLet, List[Let]]](Map())((let, c) => let match {
      case let @ VecLet(x, _) => c + (let -> lst.filter({
        case VecLet(_, v) => v.contains(x)
        case MatLet(_, m) => m.contains(x)
        case Update(_, v) => v.contains(x)
        case ExpLet(_, e) => e.contains(x) 
        case _ => false
      })) 
      case _ => c
    }) 
  }
}

// map of "let" and the list of lets that "let" depends on (parents)
//              and the list of lets that depend on "let" (children)
object def_dependency {
  def apply(lst: List[Let]) = {
    val children = scala.collection.mutable.Map[Let, Set[Let]]() 
    lst.foreach(l => children += (l -> Set[Let]()))
    
    val getParent = (let: Let) => let match {
      case VecLet(_, v) => lst.filter({
        case VecLet(x, _) => v.contains(x)
        case MatLet(x, _) => v.contains(x) 
        case _ => false
      })
      case MatLet(_, m) =>  lst.filter({
        case VecLet(x, _) => m.contains(x)
        case MatLet(x, _) => m.contains(x) 
        case _ => false
      })
      case Update(x, dx) => lst.filter({
        case VecLet(x1, v1) => dx.contains(x1) || v1.contains(x)
        case MatLet(x1, v1) => dx.contains(x1) || v1.contains(x)
        case Update(y, _) => x != y && dx.contains(y)  
        case _ => false
      })
      case ExpLet(_, e) => lst.filter({
        case VecLet(x, _) => e.contains(x) 
        case _ => false
      }) 
    } 
    
    val parents = lst.map(let => {
      val parent = getParent(let) 
      parent.foreach(p => children += (p -> children(p).+(let))) 
      (let, parent)
    }).toMap
    
    (parents, children.toMap)
  }
}

// calculate the GPU memory used by each let statement
object Memory {
  def apply(lst: List[Let]) = {
    lst.foldLeft[(List[Long], Env)]((Nil, new Env(Map())))({ case ((sizes, env), let) =>
      let match {
        case VecLet(x, v) => {
          val s =
          (if(v.isInstanceOf[VecData]) {
            0
           }
           else {
            val memorySize = size(v.getDims, env) 
            v.actIfInplace[Long](v1 => if(!v1.isInstanceOf[VecDecCopy]) 0L else memorySize, memorySize)
          })
          (sizes:::List(s), bind(x.dim, v.getDims, env))
        }
        case MatLet(x, m) => (sizes:::List(0L), env)
        case Update(_, _) => (sizes:::List(0L), env)
        case Dealloc(x) => (sizes:::List(-size(x.dim, env)), env) // find out max memory with deallocation 
        case _ => (sizes:::List(0L), env)
      }
    })
  }
  
  def size(dims: List[Dim], env: Env) = {
    dims.map(d => env(d)).map(r => r.size).reduce(_*_) * 4L
  }
  def bind(dl: List[Dim], dr: List[Dim], env: Env) = {
    (dl.zip (dr.map(d => env(d)))).foldLeft(env)((c, e) => 
      e._1 match {
        case x@DimVar(_) => c + (x, e._2)
        case _ => c
      }
     )
  }
}
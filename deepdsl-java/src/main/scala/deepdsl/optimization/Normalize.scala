package deepdsl.optimization

import deepdsl.ast._

// normalize times/plus expressions to the form of ((...((x_1, x_2), x_3)...), x_n-1), x_n)
object Normalize {
	def apply(e: Times): Times = {
	  var (r, b) = norm2(e)
	  while(b) {
		norm2(r) match {
		  case (e1, b1) => { r = e1; b = b1 }
		}
	  }
	  r
	}
	def apply(e: Plus): Plus = {
	  var (r, b) = norm2(e)
	  while(b) {
	    norm2(r) match {
	      case (e1, b1) => {r = e1; b = b1 }
	    }
	  }
	  r
	}
}

object norm2 {
	def apply(e: Times) : (Times, Boolean) = {
		e match {
		case Times(e1, Times(e2, e3)) => ( Times((Times(e1, e2)), e3), true )
		case Times(e1, e2) => {
		  
			e1 match {
			  case Times(e11, e12) => {
			    val (e4, b) = norm2(Times(e11, e12))
			    (Times(e4, e2), b)  
			  }
			  case _ => (e, false)
			}
			
		  } 
		}
	}
	def apply(e: Plus) : (Plus, Boolean) = {
	  e match {
	    case Plus(e1, Plus(e2, e3)) => (Plus (Plus(e1, e2), e3), true)
	    case Plus(e1, e2) => {
	      e1 match {
	        case Plus(e11, e12) => {
	          val (e4, b) = norm2(Plus(e11, e12))
	          (Plus(e4, e2), b)
	        }
	        case _ => (e, false)
	      }
	    }
	  }
	}
}

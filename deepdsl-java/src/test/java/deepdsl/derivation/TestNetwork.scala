package deepdsl.derivation

import deepdsl.analysis._
import deepdsl.ast.{Lmdb, Mnist, _}
import deepdsl.derivation.MemoryAnalysis._
import deepdsl.lang._
import deepdsl.layer._
import deepdsl.run._
import org.junit.Test

class TestNetwork {
  val K = 1000 // # of classes for ImageNet
  val env = new Env(Map())

  @Test
  def testResidualNN = resnet(64, 0.01f, 0.9f, 0.0005f, 1000, 10, "resnet")
  @Test
  def testVgg = vgg(64, 0.1f, 0, 0.0005f, 1000, 10, "vgg")
  @Test
  def testOverfeat = overfeat(128, 0.01f, 0.9f, 0.0005f, 1000, 10, "overfeat")
  @Test
  def testGooglenet = googlenet(128, 0.01f, 0.9f, 0.0005f, 1000, 10, "googlenet")
  @Test
  def testAlexnet = alexnet(128, 0.01f, 0.1f, 0.0005f, 1000, 10, "alexnet")
  @Test
  def testLenet = lenet(500, 0.01f, 0.1f, 0.0005f, 100, 10, "lenet")

  private def resnet(batch_size: Int, learn_rate: Float, momentum: Float, decay: Float, train_iter: Int, test_iter: Int, name: String) {
    val N = batch_size; val C = 3;  val N1 = 224; val N2 = 224 // batch size, channel, and x/y size
    val dim = List(N, C, N1, N2)
    val lmdb = Lmdb(dim, 1000000, 10000, K) // # of training images, # of test images, # of classes
    //val lmdb = Imagenet(dim, 1000000, 10000, K)  // alternative data-store for ImageNet built with Java.

    // Specifying train dataSet
    val y = T._new("Y", List(N))
    val x = T._new("X", dim)

    val relu = CudaLayer.relu(4)
    val pool = CudaLayer.max_pool(3, 2, 0)
    val ave_pool = CudaLayer.ave_pool(7, 1, 0)
    val full = Layer.full("fc", K)

    val flat = Layer.flatten(4, 1)
    val softmax = CudaLayer.log_softmax

    def cv_norm_relu(name: String, k: Int, c: Int, s: Int, p: Int) = {
      val cv = CudaLayer.convolv(name+"_cv", k, c, s, p, Param.xavier, Param.fixed(0)) // Fixed bias -- not trained
      val bn = CudaLayer.batch_norm(name+"_bn", 1f, 0f) // gamma (or scale) = 1 and beta (or bias) = 0
      val x = T._new(4)
      VecFun(x, (relu o bn o cv)(x))
    }

    def triple(name: String, m: Int, stride: Int) = {
      val x = T._new(4)
      val a = cv_norm_relu(name + "_a", 1, m*64, stride, 0)
      val b = cv_norm_relu(name + "_b", 3, m*64, 1, 1)
      val c = cv_norm_relu(name + "_c", 1, m*256, 1, 0)
      VecFun(x, (c o b o a)(x))
    }

    def branch(name: String, m: Int, stride: Int) = {
      val b1 = cv_norm_relu(name + "1", 1, m*256, stride, 0)
      val b2 = triple(name + "2", m, stride)
      val x = T._new(4)
      relu o VecFun(x, b1(x) + b2(x))
    }

    def bypass(name: String, m: Int) = {
      val x = T._new(4)

      relu o VecFun(x, triple(name, m, 1)(x) + x)
    }

    val subnet2 =  bypass("2c", 1) o bypass("2b", 1) o branch("2a", 1, 1)
    val subnet3 =  bypass("3d", 2) o bypass("3c", 2) o bypass("3b", 2) o branch("3a", 2, 2)
    val subnet4 =  bypass("4f", 4) o bypass("4e", 4) o bypass("4d", 4) o bypass("4c", 4) o bypass("4b", 4) o branch("4a", 4, 2)
    val subnet5 =  bypass("5c", 8) o bypass("5b", 8) o branch("5a", 8, 2)

    val network = full o flat o ave_pool o subnet5 o subnet4 o subnet3 o subnet2 o pool o cv_norm_relu("1", 7, 64, 2, 3)

    println(typeof(network))

    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda

    val c = Layer.loss(y1)((softmax o network)(x1))
    val p = Layer.accuracy(y, 1)(network(x1))         // top 1 accuracy

    val param = c.freeVar.toList.sortWith((a,b) => a.toString < b.toString)

    val solver = Train(name, train_iter, test_iter, learn_rate, momentum, decay, 0)
    val loop = Loop(c, p, lmdb, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    // generate training and testing file
    CudnnGen.print(loop)

    // generate forward inference file
    val inf = Inference(name, test_iter, network(x1), x, lmdb)
    CudnnGen.print(inf)
  }

  private def vgg(batch_size: Int, learn_rate: Float, momentum: Float, decay: Float, train_iter: Int, test_iter: Int, name: String) {
    val N = batch_size; val C = 3;  val N1 = 224; val N2 = 224 // batch size, channel, and x/y size

    val dim = List(N, C, N1, N2)
    val lmdb = Lmdb(dim, 1000000, 10000, K) // # of training images, # of test images, # of classes

    // Specifying train dataSet
    val y = T._new("Y", List(N))
    val x = T._new("X", dim)

    val cv11 = CudaLayer.convolv("cv11", 3, 64, 1, 1)
    val cv12 = CudaLayer.convolv("cv12", 3, 64, 1, 1)
    val cv21 = CudaLayer.convolv("cv21", 3, 128, 1, 1)
    val cv22 = CudaLayer.convolv("cv22", 3, 128, 1, 1)
    val cv31 = CudaLayer.convolv("cv31", 3, 256, 1, 1)
    val cv32 = CudaLayer.convolv("cv32", 3, 256, 1, 1)
    val cv33 = CudaLayer.convolv("cv33", 3, 256, 1, 1)
    val cv41 = CudaLayer.convolv("cv41", 3, 512, 1, 1)
    val cv42 = CudaLayer.convolv("cv42", 3, 512, 1, 1)
    val cv43 = CudaLayer.convolv("cv43", 3, 512, 1, 1)
    val cv51 = CudaLayer.convolv("cv51", 3, 512, 1, 1)
    val cv52 = CudaLayer.convolv("cv52", 3, 512, 1, 1)
    val cv53 = CudaLayer.convolv("cv53", 3, 512, 1, 1)

    val full6 = Layer.full("fc6", 4096)
    val full7 = Layer.full("fc7", 4096)
    val full8 = Layer.full("fc8", K)

    val relu = CudaLayer.relu(4)
    val relu2 = CudaLayer.relu(2)
    val pool = CudaLayer.max_pool(2)
    val drop = CudaLayer.dropout(2, 0.5f)

    val flat = Layer.flatten(4, 1)
    val softmax = CudaLayer.log_softmax

    val network =  full8 o drop o relu2 o
      full7 o drop o relu2 o
      full6 o flat o
      pool o relu o cv53 o relu o cv52 o relu o cv51 o
      pool o relu o cv43 o relu o cv42 o relu o cv41 o
      pool o relu o cv33 o relu o cv32 o relu o cv31 o
      pool o relu o cv22 o relu o cv21 o
      pool o relu o cv12 o relu o cv11

    println(typeof(network))

    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda

    val c = Layer.loss(y1)((softmax o network)(x1))
    val p = Layer.accuracy(y, 1)(network(x1))         // top 1 accuracy

    val param = c.freeVar.toList.sortWith((a,b) => a.toString < b.toString)
    val solver = Train(name, train_iter, test_iter, learn_rate, momentum, decay, 0)
    val loop = Loop(c, p, lmdb, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    // generate training and testing file
    CudnnGen.print(loop)

    // generate forward inference file
    val inf = Inference(name, test_iter, network(x1), x, lmdb)
    CudnnGen.print(inf)
  }

  private def overfeat(batch_size: Int, learn_rate: Float, momentum: Float, decay: Float, train_iter: Int, test_iter: Int, name: String) {
    val N = batch_size; val C = 3;  val N1 = 224; val N2 = 224 // batch size, channel, and x/y size

    val dim = List(N, C, N1, N2)
    val lmdb = Lmdb(dim, 1000000, 10000, K) // # of training images, # of test images, # of classes

    // Specifying train dataSet
    val y = T._new("Y", List(N))
    val x = T._new("X", dim)

    val relu = CudaLayer.relu(4)
    val pool = CudaLayer.max_pool(2, 2, 0)

    val w = Param.xavier; val b02 = Param.const(0.2f, 2, 0)

    val cv1 = CudaLayer.convolv("cv1", 11, 96, 4, 0, w, b02)
    val cv2 = CudaLayer.convolv("cv2", 5, 256, 1, 2, w, b02)
    val cv3 = CudaLayer.convolv("cv3", 3, 512, 1, 1, w, b02)
    val cv4 = CudaLayer.convolv("cv4", 3, 1024, 1, 1, w, b02)
    val cv5 = CudaLayer.convolv("cv5", 3, 1024, 1, 1, w, b02)

    val full6 = Layer.full("fc6", 3072)
    val full7 = Layer.full("fc7", 4096)
    val full8 = Layer.full("fc8", K)
    val flat = Layer.flatten(4, 1)
    val softmax = CudaLayer.log_softmax

    val network = full8 o full7 o full6 o
      flat o
      pool o relu o cv5 o
      relu o cv4 o
      relu o cv3 o
      pool o relu o cv2 o
      pool o relu o cv1

    println(typeof(network))

    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda

    val c = Layer.loss(y1)((softmax o network)(x1))
    val p = Layer.accuracy(y, 1)(network(x1))         // top 1 accuracy
    val param = c.freeVar.toList.sortWith((a,b) => a.toString < b.toString)

    val solver = Train("overfeat", train_iter, 10, learn_rate, momentum, decay, 0)
    val loop = Loop(c, p, lmdb, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    // generate training and testing file
    CudnnGen.print(loop)

    // generate forward inference file
    val inf = Inference("overfeat", test_iter, network(x1), x, lmdb)
    CudnnGen.print(inf)
  }

  private def googlenet(batch_size: Int, learn_rate: Float, momentum: Float, decay: Float, train_iter: Int, test_iter: Int, name: String) {
    val N = batch_size; val C = 3;  val N1 = 224; val N2 = 224 // batch size, channel, and x/y size

    val dim = List(N, C, N1, N2)
    val lmdb = Lmdb(dim, 1000000, 10000, K) // # of training images, # of test images, # of classes

    // Specifying train dataSet
    val y = T._new("Y", List(N))
    val x = T._new("X", dim)
    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda
    val relu = CudaLayer.relu(4)
    val relu2 = CudaLayer.relu(2)
    val lrn = CudaLayer.lrn(5, 0.0001, 0.75)
    val drop = CudaLayer.dropout(4, 0.4f)
    val drop2 = CudaLayer.dropout(2, 0.7f)
    val pool = CudaLayer.max_pool(3, 2, 1)
    val ipool = CudaLayer.max_pool(3, 1, 1)
    val pool7 = CudaLayer.ave_pool(7, 1, 0) // average pooling
    val bpool = CudaLayer.ave_pool(5, 3, 0)
    val flat = Layer.flatten(4, 1)
    val softmax_loss = Layer.loss(y1) o CudaLayer.log_softmax

    val w = Param.xavier; val b02 = Param.const(0.2f, 2, 0); val b0 = Param.const(0, 2, 0)

    val full7 = Layer.full("fc", K, w, b0)
    val cv1 = CudaLayer.convolv("cv1", 7, 64, 2, 3, w, b02)
    val cv2 = CudaLayer.convolv("cv2", 1, 64, 1, 0, w, b02)
    val cv3 = CudaLayer.convolv("cv3", 3, 192, 1, 1, w, b02)

    def inception(n: Int) = {
      val icv1 = CudaLayer.convolv(s"cv${n}1", 1, 64, 1, 0, w, b02)
      val icv2 = CudaLayer.convolv(s"cv${n}2", 1, 96, 1, 0, w, b02)
      val icv3 = CudaLayer.convolv(s"cv${n}3", 3, 128, 1, 1, w, b02)
      val icv4 = CudaLayer.convolv(s"cv${n}4", 1, 16, 1, 0, w, b02)
      val icv5 = CudaLayer.convolv(s"cv${n}5", 5, 32, 1, 2, w, b02)
      val icv6 = CudaLayer.convolv(s"cv${n}6", 1, 32, 1, 0, w, b02)

      val p = T._new(4)

      VecFun(p, CudaLayer.concat( (relu o icv1)(p),
        (relu o icv3 o relu o icv2)(p),
        (relu o icv5 o relu o icv4)(p),
        (relu o icv6 o ipool)(p) )
      )
    }

    val network3 = full7 o flat o drop o pool7 o inception(9) o inception(8) o pool o inception(7)
    val network2 = inception(6) o inception(5) o inception(4)
    val network1 = inception(3) o pool o inception(2) o inception(1) o
      pool o lrn o relu o cv3 o relu o cv2 o lrn o pool o relu o cv1

    def branch(n: Int) = {
      val cv = CudaLayer.convolv(s"b${n}cv", 1, 128, 1, 0, w, b02)
      val f1 = Layer.full(s"b${n}fc1", 1024, w, b02)
      val f2 = Layer.full(s"b${n}fc2", K, w, b0)
      f2 o drop2 o relu2 o f1 o flat o relu o cv o bpool
    }
    val stage2 = {
      val p = T._new(4)
      Vec2ScalarFun(p, (softmax_loss o network3)(p) + (softmax_loss o branch(2))(p) * Real(0.3f, "loss2"))
    }
    val stage1 = {
      val p = T._new(4)
      Vec2ScalarFun(p, (stage2 o network2)(p) + (softmax_loss o branch(1))(p) * Real(0.3f, "loss1"))
    }

    val c = (stage1 o network1)(x1)
    val p = Accuracy((network3 o network2 o network1)(x1), y, 1)         // top 1 accuracy
    val param = c.freeVar.toList.sortWith((a,b) => a.toString < b.toString)

    val solver = Train(name, train_iter, test_iter, learn_rate, momentum, decay, 0)
    val loop = Loop(c, p, lmdb, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    // generate training and testing file
    CudnnGen.print(loop)

    // generate forward inference file
    val inf = Inference(name, test_iter, (network3 o network2 o network1)(x1), x, lmdb)
    CudnnGen.print(inf)
  }

  private def alexnet(batch_size: Int, learn_rate: Float, momentum: Float, decay: Float, train_iter: Int, test_iter: Int, name: String) {
    val N = batch_size; val C = 3;  val N1 = 224; val N2 = 224 // batch size, channel, and x/y size

    val dim = List(N, C, N1, N2)
    val lmdb = Lmdb(dim, 1000000, 10000, K) // # of training images, # of test images, # of classes

    // Specifying train dataSet
    val y = T._new("Y", List(N))
    val x = T._new("X", dim)

    val relu2 = CudaLayer.relu(2)
    val relu = CudaLayer.relu(4)
    val lrn = CudaLayer.lrn(5, 0.0001, 0.75)
    val drop = CudaLayer.dropout(2, 0.5f)
    val pool = CudaLayer.max_pool(3, 2, 0)
    val flat = Layer.flatten(4, 1)
    val softmax = CudaLayer.log_softmax

    val w001 = Param.gaussian(0.01f); val w0005 = Param.gaussian(0.005f)   // std, lr_mult=1, decay_mult=1
    val b0 = Param.const(0, 2, 0); val b01 = Param.const(0.1f, 2, 0)       // value, lr_mult, decay_mult

    val cv1 = CudaLayer.convolv("cv1", 11, 96, 4, 2, w001, b0)
    val cv2 = CudaLayer.convolv("cv2", 5, 256, 1, 2, w001, b01)
    val cv3 = CudaLayer.convolv("cv3", 3, 384, 1, 1, w001, b0)
    val cv4 = CudaLayer.convolv("cv4", 3, 384, 1, 1, w001, b01)
    val cv5 = CudaLayer.convolv("cv5", 3, 256, 1, 1, w001, b01)

    val full6 = Layer.full("fc6", 4096, w0005, b01)
    val full7 = Layer.full("fc7", 4096, w0005, b01)
    val full8 = Layer.full("fc8", K, w001, b0)

    val network =       full8 o
      drop o relu2 o full7 o
      drop o relu2 o full6 o flat o
      pool o relu o cv5 o
      relu o cv4 o
      relu o cv3 o
      pool o lrn o relu o cv2 o
      pool o lrn o relu o cv1

    println(typeof(network)) // type-check the network and print out its type (a tensor function type)

    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda

    val c = Layer.loss(y1)((softmax o network)(x1))
    val p = Layer.accuracy(y, 1)(network(x1))         // top 1 accuracy
    val param = c.freeVar.toList.sortWith((a,b) => a.toString < b.toString)

    val solver = Train(name, train_iter, test_iter, learn_rate, momentum, decay, 0)
    val loop = Loop(c, p, lmdb, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    // generate training and testing file
    CudnnGen.print(loop)

    // generate forward inference file
    val inf = Inference(name, test_iter, network(x1), x, lmdb)
    CudnnGen.print(inf)
  }

  private def lenet(batch_size: Int, learn_rate: Float, momentum: Float, decay: Float, train_iter: Int, test_iter: Int, name: String) {
    val K = 10 // # of classes
    val N = batch_size; val C = 1; val N1 = 28; val N2 = 28 // batch size, channel, and x/y size

    val dim = List(N, C, N1, N2)

    // Specifying train dataSet
    val mnist = Mnist(dim)
    val y = T._new("Y", List(N))
    val x = T._new("X", dim)

    val cv1 = CudaLayer.convolv("cv1", 5, 20)
    val cv2 = CudaLayer.convolv("cv2", 5, 50)
    val mp = CudaLayer.max_pool(2)
    val flat = Layer.flatten(4, 1)
    val f = Layer.full("fc1", 500)
    val f2 = Layer.full("fc2", K)
    val softmax = CudaLayer.softmax
    val relu = CudaLayer.relu(2)

    val network = f2 o relu o f o flat o mp o cv2 o mp o cv1

    println(typeof(network))
    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda
    val c = (Layer.log_loss(y1) o softmax o network) (x1)
    val p = Layer.accuracy(y, 1)(network(x1))         // top 1 accuracy

    val param = c.freeVar.toList
    val solver = Train(name, train_iter, test_iter, learn_rate, momentum, decay, 0)

    val loop = Loop(c, p, mnist, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    // generate training and testing file
    CudnnGen.print(loop)

    // generate forward inference file
    val inf = Inference(name, test_iter, network(x1), x, mnist)
    CudnnGen.print(inf)
  }

  // This test follows the Lenet example in Theano
  @Test
  def testLenet_tanh {
    val K1 = DimConst(5); val K2 = DimConst(5) // two convolution kernel x/y dimensions
    val d = 2; val D = DimConst(d) // down-sample rate
    val K = DimConst(10) // # of classes
    val C = DimConst(1); val F1 = DimConst(20); val F2 = DimConst(50) // # of feature maps
    val N = DimConst(500); val N1 = DimConst(28); val N2 = DimConst(28) // # of images and their x/y size
    val M = N; val M1 = T.dim // fully connected layer
    val C1 = T.dim; val C2 = T.dim

    val fan_in = List(C1 * K1 * K1, C2 * K2 * K2, M1, M)
    val fan_out = List((F1 * K1 * K1)/D/D, (F2 * K2 * K2)/D/D, M, K)
    val init = (fan_in zip fan_out).map({case (x, y) => Xavier(x, y)})

    val k1 = T._new(init(0), "W1", F1, C1, K1, K1)
    val k2 = T._new(init(1), "W2", F2, C2, K2, K2)
    val w = T._new(init(2), "W", M, M1)
    val b = T._new(ConstInit(0f), "B", M)
    val b1 = T._new(ConstInit(0f), "B1", F1)
    val b2 = T._new(ConstInit(0f), "B2", F2)
    val b3 = T._new(ConstInit(0f), "B3", K)
    val theta = T._new(ConstInit(0f), "Theta", K, M)
    // Specifying train dataSet

    val dim = List(N.bound, C.bound, N1.bound, N2.bound)
    val mnist = Mnist(dim)
    val y = T._new("Y", N)
    val x = T._new("X", dim)

    val cv1 = CudaLayer.convolv(k1, b1)
    val cv2 = CudaLayer.convolv(k2, b2)
    val mp = CudaLayer.max_pool(d)
    val flat = Layer.flatten(4, 1)
    val f = Layer.full(w, b)
    val f2 = Layer.full(theta, b3)
    val softmax = CudaLayer.softmax
    val tanh2 = CudaLayer.tanh(2)
    val tanh = CudaLayer.tanh(4)

    val network = f2 o tanh2 o f o flat o  tanh o mp o cv2 o tanh o mp o cv1

    println(typeof(network))
    val x1 = x.asCuda
    val y1 = y.asIndicator(K).asCuda
    val c = Layer.log_loss(y1)((softmax o network)(x1))
    val p = Layer.accuracy(y, 1)(network(x1))         // top 1 accuracy

    val param = c.freeVar.toList
    val solver = Train("lenet_tanh", 1000, 10, 0.1f, 0f, 0f, 0)

    val loop = Loop(c, p, mnist, (x, y), param, solver)

    runtimeMemory(loop.train)
    parameterMemory(loop)
    CudnnGen.print(loop)
  }
}


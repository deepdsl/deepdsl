package deepdsl.data

import java.io.{File, FileInputStream}
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel.MapMode.READ_ONLY
import java.nio.ByteBuffer

import com.typesafe.config.{Config, ConfigFactory}

import scala.collection.JavaConversions._

trait Header {
  val header : Config
  def getHeaderLength : Int
}

case class MnistHeader(val header: Config) extends Header {
  val MAGIC = "magic"
  val NUM_ITEMS = "item"
  val SIZE_ROW = "row"
  val SIZE_COLUMN = "column"

  def getSizeMagic = header.getString(MAGIC).toInt
  def getSizeNumItems = header.getString(NUM_ITEMS).toInt
  def getSizeRow = {
    try {
      header.getString(SIZE_ROW).toInt
    } catch {
      case ex: Throwable => 0
    }
  }

  def getSizeColumn = {
    try {
      header.getString(SIZE_COLUMN).toInt
    } catch {
      case ex: Throwable => 0
    }
  }

  override def getHeaderLength = getSizeMagic + getSizeNumItems + getSizeRow + getSizeColumn

  override def toString = "sizeMagic=" + getSizeMagic + ", sizeNumItems=" + getSizeNumItems + ", sizeRow=" + getSizeRow + ", sizeColumn=" + getSizeColumn;
}

trait DataSet {

  val location: String
  val name: String
  val dataType: String
  val mode: String
  val header: Header

  var stream: FileInputStream = null
  var buffer: MappedByteBuffer = null

  def closeStream = if (stream != null) {stream.close; stream = null}

  def init = {
    if (stream == null) {
      val file = new File(location + name)
      val fileLength = file.length
      stream = new FileInputStream(file)
      buffer = stream.getChannel.map(READ_ONLY, 0, fileLength)
    }
  }

  def getImageOrLabel(num: Int, size: Int) = {
    dataType match {
      case DataLoader.IMAGE => {
        DataLoader.dataset match {
          case DataLoader.MNIST_DATASET => DataLoader.load(buffer, header.asInstanceOf[MnistHeader].getHeaderLength, size * num)
          case _ => throw new RuntimeException(DataLoader.dataset + " dataset is not supported!")
        }
      }
      case DataLoader.LABEL => {
        DataLoader.dataset match {
          case DataLoader.MNIST_DATASET =>  DataLoader.load(buffer, header.asInstanceOf[MnistHeader].getHeaderLength, num)
          case _ => throw new RuntimeException(DataLoader.dataset + " dataset is not supported!")
        }

      }
    }
  }

  def getHeaderValue : HeaderValue

  def convert(bytes: Array[Byte]): Int = ByteBuffer.wrap(bytes).getInt

  override def toString = {
    "location=" + location + ", name=" + name + ", dataType=" + dataType + ", mode=" + mode + "\nheader=[" + header + "]"
  }
}

class MnistDataSet(val location: String, val name: String, val dataType: String, val mode: String, val header: MnistHeader) extends DataSet {
  //The file pointer is always moving forward in this case (and assume getHeaderValue is called before getOneImageOrLabel, this can be improved later)
  override def getHeaderValue = {
    dataType match {
      case DataLoader.IMAGE => {
        val magicValue = super.convert(DataLoader.load(buffer, 0, header.getSizeMagic))
        val numItems = convert(DataLoader.load(buffer, header.getSizeMagic, header.getSizeNumItems))
        val numRows = convert(DataLoader.load(buffer, header.getSizeMagic + header.getSizeNumItems, header.getSizeRow))
        val numColumns = convert(DataLoader.load(buffer, header.getSizeMagic + header.getSizeNumItems + header.getSizeRow, header.getSizeColumn))
        new HeaderValue(magicValue, numItems, numRows, numColumns)
      }
      case DataLoader.LABEL => {
        val magicValue = convert(DataLoader.load(buffer, 0, header.getSizeMagic))
        val numItems = convert(DataLoader.load(buffer, header.getSizeMagic, header.getSizeNumItems))
        new HeaderValue(magicValue, numItems, 0, 0)
      }
    }
  }
}

class HeaderValue(val magicValue: Int, val numItems: Int, val numRows: Int, val numColumns: Int) {
  override def toString = "magicValue=" + magicValue + ", numItems=" + numItems + ", numRows=" + numRows + ", numColumns=" + numColumns
}

object DataLoader {
  val DATASET_NAME = "dataset_name"
  val LOCATION = "location"
  val DATASET_INFO_STR = "dataset_info"
  val TRAIN = "train"
  val TEST = "test"
  val IMAGE = "image"
  val LABEL = "label"

  val FILE_NAME = "name"
  val FILE_TYPE = "type"
  val FILE_MODE = "mode"
  val FILE_HEADER = "header"

  val MNIST_DATASET = "mnist"

  val generalConf = ConfigFactory.load
  val dataset = generalConf.getString(DATASET_NAME)
  val conf = ConfigFactory.load(dataset)
  val dataSetInfo = conf.getConfig(DATASET_INFO_STR)

  def load(buffer: MappedByteBuffer, offset: Int, length: Int) = {
    if (buffer == null) throw new RuntimeException("The buffer has not been initialized yet, please invoke init method first")
    val bytes: Array[Byte] = new Array[Byte](length)
    buffer.get(bytes, 0, length)
    bytes
  }
}

object ConfLoader {
  val PROGRESS_BAR_SIZE = 100
  var dataSet: DataSet = null
  var dataSetInfoPair : (scala.collection.mutable.Buffer[(String, String, String, String, MnistHeader)],
    scala.collection.mutable.Buffer[(String, String, String, String, MnistHeader)]) = null

  def getConf = {
    if (dataSetInfoPair == null) {
      val dataSetInfo = DataLoader.dataSetInfo
      val location = dataSetInfo.getString(DataLoader.LOCATION)
      val trainDataSetInfo = dataSetInfo.getConfigList(DataLoader.TRAIN) map (config => {
        val header = config.getConfig(DataLoader.FILE_HEADER)
        (location, config.getString(DataLoader.FILE_NAME), config.getString(DataLoader.FILE_TYPE), config.getString(DataLoader.FILE_MODE), new MnistHeader(config.getConfig(DataLoader.FILE_HEADER)))
      })
      val testDataSetInfo = dataSetInfo.getConfigList(DataLoader.TEST) map (config => {
        (location, config.getString(DataLoader.FILE_NAME), config.getString(DataLoader.FILE_TYPE), config.getString(DataLoader.FILE_MODE), new MnistHeader(config.getConfig(DataLoader.FILE_HEADER)))
      })
      dataSetInfoPair = (trainDataSetInfo, testDataSetInfo)
    }
    dataSetInfoPair
  }

  def createDataSet(params: (String, String, String, String, Header)) : DataSet = {
    params._5 match {
      case (header @ MnistHeader(_)) => new MnistDataSet(params._1, params._2, params._3, params._4, header)
      case _ => throw new RuntimeException("data set is not supported!")
    }
  }

  def getConfHeader = if (dataSet != null) dataSet.getHeaderValue else null
  def clean = if (dataSet != null) { dataSet.closeStream; dataSet = null }

  def getDataSet(isTrain: Boolean, dataType: String) = {
    //dataSet = confMap.get((if (isTrain) DataLoader.TRAIN else DataLoader.TEST) + dataType).get
    val conf = getConf
    dataSet = createDataSet(
      if (isTrain) {
        if (DataLoader.IMAGE.equals(dataType)) conf._1.get(0) else conf._1.get(1)
      } else {
        if (DataLoader.LABEL.equals(dataType)) conf._2.get(0) else conf._2.get(1)
      })
    dataSet.init
  }

  def createProgressBar(left: Int, right: Int): Unit = {
    (0 until left).map(_ => print("=")); (0 until right).map(_ => print(" "))
  }

  def getSamples(isTrain: Boolean, dataType: String, numSamples: Int, progressBarSize: Int = PROGRESS_BAR_SIZE) : List[Array[Byte]] = {
    ConfLoader.getDataSet(isTrain, dataType)
    val headerValue = ConfLoader.getConfHeader
    var samples = List[Array[Byte]]()

    val len = if (numSamples == -1 || numSamples > headerValue.numItems) headerValue.numItems else numSamples
    val numChunks = len / progressBarSize
    var curr = 0
    (0 to len - 1).toList.foreach(i => {
      if (i % numChunks == 0) {
        curr += 1; print("|" + createProgressBar(curr, progressBarSize - curr) + "|\r")
      }
      samples = samples.:+(ConfLoader.dataSet.getImageOrLabel(1, headerValue.numRows * headerValue.numColumns))
    })
    println("Done loading " + len + " " + dataType + "s")

    //clean the stream and buffer used in file reading
    ConfLoader.clean
    //Since Mnist returns unsigned byte (value 0~255) so we need to convert it as Java byte is always signed (-127 ~ 127)
    /*samples.map(sample => {
      dataType match {
        case DataLoader.IMG => sample.map (b => (b & 0xFF).toDouble)
        case DataLoader.LABEL => sample.map (b => b.toDouble)
      }
    })*/
    samples
  }
}

object DataSetMain {
  def main(args: Array[String]) {
    List(DataLoader.TRAIN, DataLoader.TEST) map (dataSetType => {
      List(DataLoader.IMAGE, DataLoader.LABEL) map (dataType => {
        ConfLoader.getDataSet((if (DataLoader.TRAIN.equals(dataSetType)) true else false), dataType)
        println(ConfLoader.dataSet)
        val headerValue = ConfLoader.getConfHeader
        println(headerValue)
        val trainImage = ConfLoader.dataSet.getImageOrLabel(1, headerValue.numRows * headerValue.numColumns)
        ConfLoader.clean
        println("The image is of size " + trainImage.length + " bytes\n")
      })
    })
  }
}
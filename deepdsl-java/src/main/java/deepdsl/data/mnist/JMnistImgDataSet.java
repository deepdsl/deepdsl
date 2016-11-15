package deepdsl.data.mnist;


public class JMnistImgDataSet extends JImgDataSet {

    private Header header;

    @Override
    public Header getHeader() {
        return header;
    }

    @Override
    public void setHeader(Integer... values) {
        this.header = new Header(values[0], values[1], values[2], values[3]);
    }

    public class Header implements IHeader {
        Integer magicValue;
        Integer numItems;
        Integer numRows;
        Integer numColumns;

        public Header(Integer magicValue, Integer numItems, Integer numRows, Integer numColumns) {
            this.magicValue = magicValue;
            this.numItems = numItems;
            this.numRows = numRows;
            this.numColumns = numColumns;
        }

        public Integer getMagicValue() {
            return magicValue;
        }

        public Integer getNumItems() {
            return numItems;
        }

        public Integer getNumRows() {
            return numRows;
        }

        public Integer getNumColumns() {
            return numColumns;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("magicValue=" + magicValue + ", numItems=" + numItems + ", numRows=" + numRows + ", numColumns=" + numColumns);
            return sb.toString();
        }
    }
}

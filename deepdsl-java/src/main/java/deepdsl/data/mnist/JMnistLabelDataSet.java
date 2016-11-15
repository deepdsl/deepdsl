package deepdsl.data.mnist;


public class JMnistLabelDataSet extends JLabelDataSet {

    private Header header;

    @Override
    public Header getHeader() {
        return header;
    }

    @Override
    public void setHeader(Integer... values) {
        this.header = new Header(values[0], values[1]);
    }

    public class Header implements IHeader {
        Integer magicValue;
        Integer numItems;

        public Header(Integer magicValue, Integer numItems) {
            this.magicValue = magicValue;
            this.numItems = numItems;
        }

        public Integer getMagicValue() {
            return magicValue;
        }

        public Integer getNumItems() {
            return numItems;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("magicValue=" + magicValue + ", numItems=" + numItems);
            return sb.toString();
        }
    }
}

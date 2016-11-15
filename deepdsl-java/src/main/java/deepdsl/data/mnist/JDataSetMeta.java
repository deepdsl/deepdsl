package deepdsl.data.mnist;

import com.google.gson.annotations.SerializedName;

public class JDataSetMeta {
    private static final String LOCATION = "location";
    private static final String TRAIN = "train";
    private static final String TEST = "test";

    @SerializedName(LOCATION) String location;
    @SerializedName(TRAIN) JTrainOrTest[] train;
    @SerializedName(TEST) JTrainOrTest[] test;

    public class JTrainOrTest {
        private static final String NAME = "name";
        private static final String TYPE = "type";
        private static final String MODE = "mode";
        private static final String HEADER = "header";

        @SerializedName(NAME) String name;
        @SerializedName(TYPE) String dataType;
        @SerializedName(MODE) String mode;
        @SerializedName(HEADER) Header header;

        public String getName() {
            return name;
        }

        public String getDataType() {
            return dataType;
        }

        public String getMode() {
            return mode;
        }

          public Header getHeader() {
            return header;
        }

        public String toString(int precedent) {
            StringBuilder sb = new StringBuilder();
            String str = "name=" + name + ", dataType=" + dataType + ", mode=" + mode + ", header=\n";
            return sb.append(str).append(JDataSetMeta.getIndent(precedent + 2 + str.length()) + header.toString(precedent + str.length())).toString();
        }
    }

    public class Header {
        private static final String MAGIC = "magic";
        private static final String ITEM = "item";
        private static final String ROW = "row";
        private static final String COLUMN = "column";

        @SerializedName(MAGIC) Integer magic;
        @SerializedName(ITEM) Integer item;
        @SerializedName(ROW) Integer row;
        @SerializedName(COLUMN) Integer column;

        public Integer getMagic() {
            return magic;
        }

        public Integer getItem() {
            return item;
        }

        public Integer getRow() {
            return row;
        }

        public Integer getColumn() {
            return column;
        }

        public String toString(int precedent) {
            StringBuilder sb = new StringBuilder();
            String str = "magic=" + magic + ", item=" + item + ", row=" + row + ", column=" + column;
            return sb.append(str + "\n").toString();
        }
    }

    public String getLocation() {
        return location;
    }

    public JTrainOrTest[] getTrain() {
        return train;
    }

    public JTrainOrTest[] getTest() {
        return test;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("location=" + location + "\n");
        String firstPart = ", train=\n";
        sb.append(firstPart);
        for (JTrainOrTest trainElem : train) {
            sb.append(getIndent(firstPart.length()) + trainElem.toString(firstPart.length()) + "\n");
        }
        String secondPart = ", test=\n";
        sb.append(secondPart);
        for (JTrainOrTest testElem : test) {
            sb.append(getIndent(secondPart.length()) + testElem.toString(secondPart.length()));
        }
        return sb.toString();
    }

    private static String getIndent(int len) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len - 1; i++) {
            sb.append(" ");
        }
        return sb.append("|->").toString();
    }
}

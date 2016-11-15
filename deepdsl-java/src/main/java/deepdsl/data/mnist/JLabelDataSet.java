package deepdsl.data.mnist;

import java.util.ArrayList;
import java.util.List;

public abstract class JLabelDataSet {

    private final List<Integer> intLabels = new ArrayList<>();

    private final List<Byte> labels = new ArrayList<>();

    public void convertByteListToIntList() {
        if ((labels.size() == 0) && intLabels.size() > 0) {
            for (byte label : labels) {
                intLabels.add(new Integer(label & 0xFF));
            }
            labels.clear();
        }
    }

    public List<Integer> getIntLabels() {
        return intLabels;
    }

    public List<Byte> getLabels() {
        return labels;
    }

    public void addLabel(byte label) {
        labels.add(label);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("" + getHeader() + ", label.number=" + labels.size() + ", leading 0-5 label data=\n");
        for (int i = 0; i < (5 <= labels.size() ? 5 : labels.size()); i++) {
            int label = labels.get(i);
            int returnCheck = 0;
            sb.append(label + " ");
            if (++returnCheck % 16 == 0) {returnCheck = 0; sb.append("\n");}
        }
        return sb.toString();
    }

    public abstract IHeader getHeader();

    public abstract void setHeader(Integer... values);
}

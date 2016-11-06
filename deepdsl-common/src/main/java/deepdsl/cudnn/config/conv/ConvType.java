package deepdsl.cudnn.config.conv;

public enum ConvType {
	OneD(1), TwoD(2);

	private final int tpe;

	public int tpe() {
		return tpe;
	}

	ConvType(int tpe) {
		this.tpe = tpe;
	}
}
package deepdsl.cudnn.config;

public enum ConvolutionType {
	OneD(1), TwoD(2);

	private final int tpe;

	public int value() {
		return tpe;
	}

	ConvolutionType(int tpe) {
		this.tpe = tpe;
	}
}
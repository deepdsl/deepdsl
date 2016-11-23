package deepdsl.data.imagenet;

import com.google.protobuf.InvalidProtocolBufferException;

import org.fusesource.lmdbjni.*;

import java.util.Iterator;
import java.util.Locale;

import static org.fusesource.lmdbjni.Constants.bytes;
import static org.fusesource.lmdbjni.Constants.string;

public class LmdbUtils {
    enum OS {WINDOWS, LINUX, OSX}
    
    static OS os = operating_system();
    
	static OS operating_system() {
		OS os;
		
		String name = System.getProperty("os.name").toLowerCase(Locale.ENGLISH);
		if(name.contains("win")) {
			os = OS.WINDOWS;
		}
		else if (name.contains("nux")) {
			os = OS.LINUX;
		}
		else if (name.contains("mac")) {
			os = OS.OSX;
		}
		else {
			throw new RuntimeException("unsupported operation system for Lmdb code: " + name);
		}
		return os;
	}

    /**
     * This method loop through lmdb EntryIterator to get numSamples image byte array data
     *
     * @param tx the live transaction that is used to control all lmdb read/write access
     * @param db the db handler that is used to iterate through the memory map data of lmdb
     * @param numSamples the number of samples that will be read out from lmdb
     * @param keepTx whether or not the caller wants to keep the transaction alive, if false, then the lmdb transaction will be aborted at the end of this call
     *               and nor further opearations can be done with the given transaction
     * @return the byte array that is formed by concatenating all the image byte array data
     */
    public static byte[] getImageAsByteArray(Transaction tx, Database db, int numSamples, boolean keepTx) {
        byte[] a = new byte[numSamples * getSize(tx, db)];
        int i = 0;
        Iterable<Entry> iterable = getIterable(tx, db);
        for (Entry next : iterable) {
            try {
                byte[] src;
                switch(os) {
                case OSX:       
                    ImagenetProtoOSX.Datum datum = ImagenetProtoOSX.Datum.parseFrom(next.getValue());
                    System.out.println("label = " + datum.getLabel());
                    src = datum.getData().toByteArray();
                    break;
                case LINUX:
                    ImagenetProtoLinux.Datum datum2 = ImagenetProtoLinux.Datum.parseFrom(next.getValue());
                    src = datum2.getData().toByteArray();
                    break;
                case WINDOWS:
                    ImagenetProtoWin32.Datum datum3 = ImagenetProtoWin32.Datum.parseFrom(next.getValue());
                    src = datum3.getData().toByteArray();
                    break;
                default:
                	throw new RuntimeException("The operation system is not supported");
                }
                System.arraycopy(src, 0, a, i++ * src.length, src.length);
            } catch (InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
            }
            if (i == numSamples) break;
        }
        if (!keepTx) abortTransaction(tx);
        return a;
    }

    public static Iterable<Entry> getIterable(Transaction tx, Database db) {
        EntryIterator it = db.iterate(tx);
        return it.iterable();
    }

    public static Iterator<Entry> getIterator(Iterable<Entry> iterable) {
        return iterable.iterator();
    }

    public static Tuple nextTuple(Entry entry) {
        byte[] image;
        int label;
        int[] dims = new int[3];
        try {
        	switch(os) {
        	case OSX:
                ImagenetProtoOSX.Datum datum = ImagenetProtoOSX.Datum.parseFrom(entry.getValue());
                image = datum.getData().toByteArray();
                label = datum.getLabel();
                dims[0] = datum.getChannels(); dims[1] = datum.getHeight(); dims[2] = datum.getWidth();
                break;
        	case LINUX:
                ImagenetProtoLinux.Datum datum2 = ImagenetProtoLinux.Datum.parseFrom(entry.getValue());
                image = datum2.getData().toByteArray();
                label = datum2.getLabel();
                dims[0] = datum2.getChannels(); dims[1] = datum2.getHeight(); dims[2] = datum2.getWidth();
                break;
            case WINDOWS:
                ImagenetProtoWin32.Datum datum3 = ImagenetProtoWin32.Datum.parseFrom(entry.getValue());
                image = datum3.getData().toByteArray();
                label = datum3.getLabel();
                dims[0] = datum3.getChannels(); dims[1] = datum3.getHeight(); dims[2] = datum3.getWidth();
                break;
            default: throw new RuntimeException("The operation system is not supported");
        	}
        } catch (InvalidProtocolBufferException e) {
            throw new RuntimeException(e);
        }
        return new Tuple(image, label, dims);
    }


    public static Env getEnv(String path) {
        return new Env(path);
    }

    public static void abortTransaction(Transaction tx) {
        tx.abort();
    }

    /**
     * The below just demos how to write key-values to the lmdb
     *
     * @param db
     * @param key
     * @param value
     */
    public static void writeKVPair(Database db, String key, String value) {
        try {
            db.put(bytes(key), bytes(value));
            String v = string(db.get(bytes(key)));
            System.out.printf("value=" + v);
            db.delete(bytes(key));
        } catch(Exception e) {
            System.out.println(e.getMessage());
        }
    }

    public static String getValue(Database db, String key) {
        try {
            return string(db.get(bytes(key)));
        } catch(Exception e) {
            return null;
        }
    }

    public static void deleteKV(Database db, String key) {
        try {
            db.delete(bytes(key));
        } catch(Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    public static int getSize(Transaction tx, Database db) {
    	int[] dims = getDims(tx, db);
    	return dims[0] * dims[1] * dims[2];
    }

    public static int[] getDims(Transaction tx, Database db) {
        try(BufferCursor cursor = db.bufferCursor(tx)) {
            if (cursor.first()) {
                try {
                	switch(os) {
                	case OSX: 
                        ImagenetProtoOSX.Datum datum = ImagenetProtoOSX.Datum.parseFrom(cursor.valBytes());
                        return new int[]{datum.getChannels(), datum.getHeight(), datum.getWidth()};
                	case LINUX:
                        ImagenetProtoLinux.Datum datum2 = ImagenetProtoLinux.Datum.parseFrom(cursor.valBytes());
                        return new int[]{datum2.getChannels(), datum2.getHeight(), datum2.getWidth()};
                	case WINDOWS:
                        ImagenetProtoWin32.Datum datum3 = ImagenetProtoWin32.Datum.parseFrom(cursor.valBytes());
                        return new int[]{datum3.getChannels(), datum3.getHeight(), datum3.getWidth()};
                    default: throw new RuntimeException("The operation system is not supported");
                	}
                } catch (InvalidProtocolBufferException e) {
                    throw new RuntimeException(e);
                }
            }
            else throw new RuntimeException("Failed to position cursor to the first key");
        }
    }

    public static void main(String[] argv) {
        Env env = getEnv("dataset/imagenet/ilsvrc12_train_lmdb");
        Transaction tx = env.createReadTransaction();
        Database db = env.openDatabase();
        getImageAsByteArray(tx, db, 20, true);
        //The below just demos how to get iterator
        getIterator(getIterable(tx, db));
        abortTransaction(tx);
    }
}

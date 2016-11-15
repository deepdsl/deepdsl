package deepdsl.data.imagenet;

import com.google.protobuf.InvalidProtocolBufferException;
import org.fusesource.lmdbjni.*;

import java.util.Iterator;

import static org.fusesource.lmdbjni.Constants.bytes;
import static org.fusesource.lmdbjni.Constants.string;

public class LmdbUtils {

    public enum OS {
        OSX("osx"), LINUX("linux"), WINDOWS("win32");
        private final String tpe;
        public String tpe() {
            return tpe;
        }
        OS(String tpe) {
            this.tpe = tpe;
        }
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
    public static byte[] getImageAsByteArray(Transaction tx, Database db, int numSamples, boolean keepTx, OS os) {
        byte[] a = new byte[numSamples * getSize(tx, db, os)];
        int i = 0;
        Iterable<Entry> iterable = getIterable(tx, db);
        for (Entry next : iterable) {
            try {
                byte[] src;
                if (os == OS.OSX) {
                    ImagenetProtoOSX.Datum datum = ImagenetProtoOSX.Datum.parseFrom(next.getValue());
                    System.out.println("label = " + datum.getLabel());
                    src = datum.getData().toByteArray();
                } else if (os == OS.LINUX) {
                    ImagenetProtoLinux.Datum datum = ImagenetProtoLinux.Datum.parseFrom(next.getValue());
                    src = datum.getData().toByteArray();
                } else if (os == OS.WINDOWS) {
                    ImagenetProtoWin32.Datum datum = ImagenetProtoWin32.Datum.parseFrom(next.getValue());
                    src = datum.getData().toByteArray();
                } else throw new RuntimeException("The operation system is not supported");
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

    public static Tuple nextTuple(Entry entry, OS os) {
        byte[] image;
        int label;
        int[] dims = new int[3];
        try {
            if (os == OS.OSX) {
                ImagenetProtoOSX.Datum datum = ImagenetProtoOSX.Datum.parseFrom(entry.getValue());
                image = datum.getData().toByteArray();
                label = datum.getLabel();
                dims[0] = datum.getChannels(); dims[1] = datum.getHeight(); dims[2] = datum.getWidth();
            } else if (os == OS.LINUX) {
                ImagenetProtoLinux.Datum datum = ImagenetProtoLinux.Datum.parseFrom(entry.getValue());
                image = datum.getData().toByteArray();
                label = datum.getLabel();
                dims[0] = datum.getChannels(); dims[1] = datum.getHeight(); dims[2] = datum.getWidth();
            } else if (os == OS.WINDOWS) {
                ImagenetProtoWin32.Datum datum = ImagenetProtoWin32.Datum.parseFrom(entry.getValue());
                image = datum.getData().toByteArray();
                label = datum.getLabel();
                dims[0] = datum.getChannels(); dims[1] = datum.getHeight(); dims[2] = datum.getWidth();
            } else throw new RuntimeException("The operation system is not supported");
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
    
    public static int getSize(Transaction tx, Database db, OS os) {
    	int[] dims = getDims(tx, db, os);
    	return dims[0] * dims[1] * dims[2];
    }

    public static int[] getDims(Transaction tx, Database db, OS os) {
        try(BufferCursor cursor = db.bufferCursor(tx)) {
            if (cursor.first()) {
                try {
                    if (os == OS.OSX) {
                        ImagenetProtoOSX.Datum datum = ImagenetProtoOSX.Datum.parseFrom(cursor.valBytes());
                        return new int[]{datum.getChannels(), datum.getHeight(), datum.getWidth()};
                    } else if (os == OS.LINUX) {
                        ImagenetProtoLinux.Datum datum = ImagenetProtoLinux.Datum.parseFrom(cursor.valBytes());
                        return new int[]{datum.getChannels(), datum.getHeight(), datum.getWidth()};
                    } else if (os == OS.WINDOWS) {
                        ImagenetProtoWin32.Datum datum = ImagenetProtoWin32.Datum.parseFrom(cursor.valBytes());
                        return new int[]{datum.getChannels(), datum.getHeight(), datum.getWidth()};
                    } else throw new RuntimeException("The operation system is not supported");
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
        getImageAsByteArray(tx, db, 20, true, OS.OSX);
        //The below just demos how to get iterator
        getIterator(getIterable(tx, db));
        abortTransaction(tx);
    }
}

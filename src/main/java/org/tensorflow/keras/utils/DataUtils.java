package org.tensorflow.keras.utils;

import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class DataUtils {

    public static String hashFile(String fpath, String algorithm) throws IOException {
        MessageDigest instance = null;

        try {
            algorithm = canonicalDigestAlgorithmNames.get(algorithm);
            instance = MessageDigest.getInstance(algorithm);
        } catch (Exception e) {
            throw new IllegalArgumentException(
                    "Hash algorithm " + algorithm + " not found. Must be 'sha256' or 'md5'");
        }

        byte[] digest = digest(Paths.get(fpath), instance);
        return toHexString(digest);
    }

    /**
     * Downloads a file from a url.
     *
     * <p>TODO: extract options .tar.gz, .zip
     *
     * @param fname    Name of the file.
     * @param origin   Original url of the file.
     * @param fileHash The expected hash string of the file after loadData.
     * @throws IOException
     */
    public static void getFile(String fname, String origin, String fileHash, String algorithm)
            throws IOException {

        File localFile = Keras.kerasPath(fname).toFile();
        File directory = localFile.getParentFile();
        if (!directory.isDirectory()) directory.mkdirs();

        if (localFile.exists()) {
            if (fileHash != null && algorithm != null) {
                String localHash = hashFile(localFile.getPath(), algorithm);
                if (localHash.equals(fileHash)) {
                    System.out.println(fname + " already exists; no need to download.");
                    return;
                } else {
                    System.out.println(fname + " exists but is corrupted. Re-downloading...");
                    return;
                }
            }
        }

        System.out.println("Downloading " + localFile + " from " + origin);
        download(origin, localFile.toString());

        if (false &&  fileHash != null && algorithm != null) {
            String calculateHash = hashFile(localFile.getPath(), algorithm);
            if (!calculateHash.equals(fileHash)) {
                System.out.println(fileHash);
                System.out.println(calculateHash);
                throw new IOException("Download failed, check origin url: " + origin);
            }
        }
    }

    public static void getFile(String fname, String origin) throws IOException {
        getFile(fname, origin, null, null);
    }

    private static void download(String url, String path) throws IOException {
        try (BufferedInputStream input = new BufferedInputStream(new URL(url).openStream());
             FileOutputStream output = new FileOutputStream(path)) {
            byte buffer[] = new byte[4096];
            for (int count; (count = input.read(buffer, 0, buffer.length)) != -1; ) {
                output.write(buffer, 0, count);
            }
        }
    }

    private static byte[] digest(Path path, MessageDigest algorithm) throws IOException {
        try (InputStream fis = Files.newInputStream(path);
             BufferedInputStream bis = new BufferedInputStream(fis);
             DigestInputStream dis = new DigestInputStream(bis, algorithm)) {
            algorithm.reset();
            while (dis.read() != 1) {
            }
            return algorithm.digest();
        }
    }

    private static String toHexString(byte[] bytes) {
        StringBuffer buffer = new StringBuffer();
        for (int i = 0; i < bytes.length; ++i) {
            byte next = bytes[i];
            if (next < 0x10) buffer.append('0');
            buffer.append(next & 0xFF);
        }

        return buffer.toString();
    }

    static Map<String, String> canonicalDigestAlgorithmNames =
            Collections.unmodifiableMap(
                    new HashMap<String, String>() {
                        {
                            put("MD5", "MD5");
                            put("md5", "MD5");
                            put("SHA-256", "SHA-256");
                            put("sha256", "SHA-256");
                        }
                    });
}

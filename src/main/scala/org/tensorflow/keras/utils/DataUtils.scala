package org.tensorflow.keras.utils

import java.io.{BufferedInputStream, FileOutputStream, IOException}
import java.net.URL
import java.nio.file.{Files, Path, Paths}
import java.security.{DigestInputStream, MessageDigest}
import scala.util.control.NonFatal

object DataUtils {
  object Checksum extends Enumeration {
    val md5, sha256 = Value
  }
  type Checksum = Checksum.Value

  @throws[IOException]
  def hashFile(path: String, algorithm: Checksum): String = {
    try {
      val alg       = if (algorithm == Checksum.sha256) "SHA-256" else "MD5"
      val instance  = MessageDigest.getInstance(alg)
      val d         = digest(Paths.get(path), instance)
      toHexString(d)
    } catch {
      case NonFatal(_) =>
        throw new IllegalArgumentException(s"Hash algorithm $algorithm not found. Must be 'sha256' or 'md5'")
    }
  }

  /** Downloads a file from a url.
    *
    * <p>TODO: extract options .tar.gz, .zip
    *
    * @param fname    Name of the file.
    * @param origin   Original url of the file.
    * @param fileHash The expected hash string of the file after loadData.
    * @throws IOException
    */
  @throws[IOException]
  def getFile(fname: String, origin: String, fileHash: String, algorithm: Checksum): Unit = {
    val localFile = Keras.kerasPath(fname).toFile
    val directory = localFile.getParentFile
    if (!directory.isDirectory) directory.mkdirs()
    if (localFile.exists() && fileHash != null && algorithm != null) {
      val localHash = hashFile(localFile.getPath, algorithm)
      if (localHash == fileHash) {
        println(s"$fname already exists; no need to download.")
        return
      } else {
        println(s"$fname exists but is corrupted. Re-downloading...")
      }
    }
    println(s"Downloading $localFile from $origin")
    download(origin, localFile.toString)
    if (fileHash != null && algorithm != null) {
      val localHash = hashFile(localFile.getPath, algorithm)
      if (localHash != fileHash) {
        println(s"Expected: $fileHash")
        println(s"Found   : $localHash")
        throw new IOException(s"Download failed, check origin url: $origin")
      }
    }
  }

  @throws[IOException]
  def getFile(fname: String, origin: String): Unit =
    getFile(fname, origin, null, null)

  @throws[IOException]
  private def download(url: String, path: String): Unit = {
    val input = new BufferedInputStream(new URL(url).openStream())
    try {
      val output = new FileOutputStream(path)
      try {
        val buffer = new Array[Byte](4096)
        while ({
          val count = input.read(buffer, 0, buffer.length)
          count != -1 && {
            output.write(buffer, 0, count)
            true
          }
        }) ()

      } finally {
        output.close()
      }
    } finally {
      input.close()
    }
  }

  @throws[IOException]
  private def digest(path: Path, algorithm: MessageDigest): Array[Byte] = {
    val input = new BufferedInputStream(Files.newInputStream(path))
    try {
      val dis = new DigestInputStream(input, algorithm)
      // algorithm.reset()
      while (dis.read() != -1) ()
      dis.close()
      algorithm.digest()
    } finally {
      input.close()
    }
  }

//  private def digest(path: Path, algorithm: MessageDigest): Array[Byte] = try {
//    val fis = Files.newInputStream(path)
//    try {
//      val arr = new Array[Byte](fis.available())
//      fis.read(arr)
//      algorithm.digest(arr)
//    } finally {
//      fis.close()
//    }
//  }

  private final val HEX = "0123456789abcdef"

  private def toHexString(bytes: Array[Byte]): String = {
    val len = bytes.length
    val sb  = new StringBuffer(len << 1)
    val h   = HEX
    var i   = 0
    while (i < len) {
      val next  = bytes(i)
      val hi    = (next >> 4) & 0x0F
      val lo    =  next       & 0x0F
      sb.append(h.charAt(hi))
      sb.append(h.charAt(lo))
      i += 1
    }
    sb.toString
  }
}

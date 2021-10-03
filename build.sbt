lazy val projectVersion = "0.1.0-SNAPSHOT"

lazy val deps = new {
  val main = new {
    val tensorflow      = "0.3.3"
  }
  val test = new {
    val junitInterface  = "0.13.2"
    val junitJupiter    = "5.8.1"
  }
}

// change this according to your OS and CPU
val tfClassifier = "linux-x86_64"

lazy val root = project.in(file("."))
  .settings(
    name             := "tensorflow-keras",
    version          := projectVersion,
    organization     := "de.sciss",
    licenses         := Seq("LGPL v2.1+" -> url("http://www.gnu.org/licenses/lgpl-2.1.txt")),
    scalaVersion     := "2.13.6",
    crossPaths       := false,
    autoScalaLibrary := false,
    javacOptions    ++= Seq("-Xlint:unchecked"),
    libraryDependencies ++= Seq(
      "org.tensorflow"  %  "tensorflow-core-api"    % deps.main.tensorflow,
      "org.tensorflow"  %  "tensorflow-core-api"    % deps.main.tensorflow classifier tfClassifier, // "linux-x86_64-mkl",
      "org.tensorflow"  %  "tensorflow-framework"   % deps.main.tensorflow,
      "org.junit.jupiter" % "junit-jupiter-engine"  % deps.test.junitJupiter    % Test,
      "com.github.sbt"  % "junit-interface"         % deps.test.junitInterface  % Test
    )
  )

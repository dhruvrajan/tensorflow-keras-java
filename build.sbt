lazy val baseName       = "tensorflow-keras"
lazy val projectVersion = "0.1.0-SNAPSHOT"

// ---- dependencies ----

lazy val deps = new {
  val main = new {
    val tensorflow      = "0.4.0"
  }
  val test = new {
    val junitInterface  = "0.13.2"
    val junitJupiter    = "5.8.2"
  }
}

// change this according to your OS and CPU
val tfClassifier = "linux-x86_64"

ThisBuild / version       := projectVersion
ThisBuild / organization  := "de.sciss"
ThisBuild / versionScheme := Some("pvp")

// ---- project ----

def basicJavaOpts = Seq("-source", "1.8")

lazy val root = project.in(file("."))
  .settings(publishSettings)
  .settings(
    name             := baseName,
    licenses         := Seq("LGPL v2.1+" -> url("http://www.gnu.org/licenses/lgpl-2.1.txt")),
    scalaVersion     := "2.13.7",
    crossScalaVersions := Seq("3.1.0", "2.13.7", "2.12.15"),
    scalacOptions   ++= Seq("-deprecation"),
    javacOptions                 := basicJavaOpts ++ Seq("-Xlint:unchecked", "-target", "1.8"),
    Compile / doc / javacOptions := basicJavaOpts,
    libraryDependencies ++= Seq(
      "org.tensorflow"  %  "tensorflow-core-api"    % deps.main.tensorflow,
      "org.tensorflow"  %  "tensorflow-core-api"    % deps.main.tensorflow classifier tfClassifier, // "linux-x86_64-mkl",
      "org.tensorflow"  %  "tensorflow-framework"   % deps.main.tensorflow,
      "org.junit.jupiter" % "junit-jupiter-engine"  % deps.test.junitJupiter    % Test,
      "com.github.sbt"  % "junit-interface"         % deps.test.junitInterface  % Test
    )
  )

// ---- publishing ----

lazy val publishSettings = Seq(
  publishMavenStyle := true,
  Test / publishArtifact := false,
  pomIncludeRepository := { _ => false },
  developers := List(
    Developer(
      id    = "dhruvrajan",
      name  = "Dhruv Rajan",
      email = "dhruv@krishnaprem.com",
      url   = url("https://github.com/dhruvrajan")
    ),
    Developer(
      id    = "sciss",
      name  = "Hanns Holger Rutz",
      email = "contact@sciss.de",
      url   = url("https://www.sciss.de")
    ),
  ),
  scmInfo := {
    val h = "github.com"
    val a = s"Sciss/$baseName"
    Some(ScmInfo(url(s"https://$h/$a"), s"scm:git@$h:$a.git"))
  },
)

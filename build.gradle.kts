plugins {
    kotlin("jvm") version "1.9.0"
    application
}

group = "test"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

application {
    mainClass = "ApplicationKt"
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime:1.16.1")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    jvmToolchain(17)
}
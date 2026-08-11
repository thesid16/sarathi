plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "in.sarathi.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "in.sarathi.app"
        // API 26 is the floor for a foreground service done properly, and
        // reaches essentially every phone still in use in the target market.
        minSdk = 26
        targetSdk = 36
        versionCode = 2
        versionName = "0.2.0"

        // Phones only. x86 and x86_64 are 38.6 MB of the APK and exist for
        // emulators; no Android phone in the target market runs them. Dropping
        // them takes the download from 92 MB to 53 MB, which matters when the
        // people receiving it are on metered connections.
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
    }

    // Signed with a key committed to this repository, on purpose.
    //
    // Sarathi is not distributed through a store, and the people it is for are
    // handed an APK by a teacher, an NGO or a friend. An unsigned build simply
    // will not install, and "generate your own keystore first" is a barrier
    // that stops the handover entirely. The key proves nothing about identity
    // here - anyone can rebuild - and its only job is to satisfy the installer
    // and let updates replace the previous version rather than fail with a
    // signature mismatch.
    //
    // Anyone shipping this seriously should replace it and keep it private.
    signingConfigs {
        create("public") {
            storeFile = file("../sarathi-release.jks")
            storePassword = "sarathi-public"
            keyAlias = "sarathi"
            keyPassword = "sarathi-public"
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("public")
            // Shrinking is off. R8 strips classes that are only reached
            // reflectively, and this app resolves adapters and YAML types that
            // way; a stripped release that crashes on a model the debug build
            // handles is exactly the failure this project keeps having to hunt,
            // and the APK is dominated by native libraries R8 cannot touch
            // anyway. Correct and 96 MB beats clever and broken.
            isMinifyEnabled = false
            isShrinkResources = false
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    // Kotlin 2.3 removed the kotlinOptions DSL. Forced here rather than
    // chosen: LiteRT-LM 0.15.0 ships Kotlin 2.3 metadata, which a 2.1 compiler
    // refuses to read at all.
    kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
    buildFeatures { viewBinding = false }

    // Model files are already compressed; letting the packager re-compress
    // them costs build time and, worse, forces a decompress-to-disk on first
    // run before the interpreter can mmap them.
    androidResources { noCompress += listOf("tflite", "litertlm", "onnx") }
}

// Copy the shared data into assets at build time rather than keeping a second
// copy in the repo. Model manifests, phrase tables and the label set are the
// contract between the prototype and the app; duplicating them is how the two
// drift apart and how a model benchmarked on a laptop stops meaning anything
// on the phone. If a file is missing the build fails here rather than the app
// failing at runtime on someone's walk.
val sharedRoot = rootProject.file("..")

val copySharedAssets by tasks.registering(Copy::class) {
    from(File(sharedRoot, "models/manifests")) { into("manifests") }
    from(File(sharedRoot, "models/labels")) { into("labels") }
    from(File(sharedRoot, "phrases")) { into("phrases") }
    from(File(sharedRoot, "training/taxonomy")) {
        include("sarathi77.yaml", "size_priors.yaml", "coco_to_sarathi.yaml")
        into("taxonomy")
    }
    into(layout.buildDirectory.dir("generated/sharedAssets"))
}

android.sourceSets.getByName("main").assets.srcDir(
    layout.buildDirectory.dir("generated/sharedAssets")
)
tasks.named("preBuild") { dependsOn(copySharedAssets) }

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.lifecycle:lifecycle-service:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    val camerax = "1.4.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    // PreviewView, for the live feed on screen.
    implementation("androidx.camera:camera-view:$camerax")

    // LiteRT (formerly TFLite). GPU delegate is a separate artifact.
    implementation("com.google.ai.edge.litert:litert:1.2.0")
    implementation("com.google.ai.edge.litert:litert-gpu:1.2.0")

    // LiteRT-LM, for on-demand scene description with Gemma 4. Pinned rather
    // than `latest.release`: this project publishes measured numbers, and a
    // number is meaningless if the runtime that produced it can change under
    // the build.
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.15.0")

    // ML Kit text recognition, UNBUNDLED. These artifacts ship no model in the
    // APK - recognition happens in Google Play Services, the same place the
    // TextToSpeech this app already depends on lives. See the licence
    // reasoning at the top of ocr/TextReader.kt; OCR is optional and the app
    // works without it.
    //
    // Devanagari is a separate model, not an option on the Latin one. A Hindi
    // sign read by the Latin recogniser returns nothing, which is
    // indistinguishable from "no text here".
    implementation("com.google.android.gms:play-services-mlkit-text-recognition:19.0.1")
    implementation("com.google.android.gms:play-services-mlkit-text-recognition-devanagari:16.0.1")

    // YAML, so the phone reads the same manifests and phrase tables the
    // prototype does rather than a hand-maintained Kotlin copy.
    implementation("org.yaml:snakeyaml:2.3")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

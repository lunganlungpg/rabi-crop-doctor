plugins {
    id("com.android.application")
}

android {
    namespace = "in.iasri.rabidoctor"
    compileSdk = 36

    defaultConfig {
        applicationId = "in.iasri.rabidoctor"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0.0"
    }

    signingConfigs {
        create("debugConfig") {
            storeFile = file("${rootDir}/debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("debugConfig")
        }
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debugConfig")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    sourceSets {
        getByName("main") {
            manifest.srcFile("src/main/AndroidManifest.xml")
            java.setSrcDirs(listOf("../android_build/src"))
            res.setSrcDirs(listOf("../android_build/res"))
            assets.setSrcDirs(listOf("../android_build/assets"))
        }
    }
}

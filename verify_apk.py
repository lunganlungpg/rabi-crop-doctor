import os
import subprocess
import zipfile

import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP1_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
APK_PATH = os.path.join(BASE_DIR, "RabiCropDoctor.apk")
SDK_DIR = r"C:\Users\lungh\AppData\Local\Android\Sdk"
BUILD_TOOLS = os.path.join(SDK_DIR, "build-tools", "35.0.1")
AAPT = os.path.join(BUILD_TOOLS, "aapt.exe")
APKSIGNER = os.path.join(BUILD_TOOLS, "apksigner.bat")


def _resolve_jdk_dir():
    for cand in [
        os.environ.get("JAVA_HOME", ""),
        os.path.join(APP1_DIR, "jdk-17", "jdk-17.0.12+7"),
        r"C:\Users\lungh\.p2\pool\plugins\org.eclipse.justj.openjdk.hotspot.jre.full.win32.x86_64_21.0.10.v20260205-0638\jre",
    ]:
        if cand and os.path.exists(os.path.join(cand, "bin", "java.exe")):
            return cand
    java = shutil.which("java")
    return os.path.dirname(os.path.dirname(java)) if java else os.path.join(APP1_DIR, "jdk-17", "jdk-17.0.12+7")


JDK_DIR = _resolve_jdk_dir()


def main():
    print("=" * 65)
    print("RABI CROP DOCTOR APK VERIFICATION SUITE")
    print("=" * 65)

    assert os.path.exists(APK_PATH), f"APK not found at {APK_PATH}"
    size_kb = os.path.getsize(APK_PATH) / 1024.0
    print(f"APK Path: {APK_PATH}")
    print(f"APK Size: {size_kb:.2f} KB ({os.path.getsize(APK_PATH):,} bytes)")

    required = [
        "AndroidManifest.xml",
        "classes.dex",
        "resources.arsc",
        "assets/index.html",
        "assets/styles/app.css",
        "assets/js/feature_extractor.js",
        "assets/js/inference_engine.js",
        "assets/js/models_data.js",
        "assets/js/ai_vision.js",
        "assets/js/app.js",
        "res/drawable/ic_launcher.xml",
        "res/mipmap-anydpi-v26/ic_launcher.xml",
    ]
    print("\n[1] Checking embedded offline ML/ANN web assets & Android entries:")
    with zipfile.ZipFile(APK_PATH, "r") as z:
        names = set(z.namelist())
        for req in required:
            assert req in names, f"Missing required APK entry: {req}"
            print(f"  - {req:<38}: FOUND")

    print("\n[2] Checking AAPT Manifest & Badging:")
    res = subprocess.run([AAPT, "dump", "badging", APK_PATH], capture_output=True, text=True, check=True)
    for line in res.stdout.splitlines():
        if any(k in line for k in ("package:", "sdkVersion:", "targetSdkVersion:", "application-label:", "launchable-activity:", "uses-permission:")):
            print(f"  {line}")

    print("\n[3] Verifying APK cryptographic signature:")
    env = os.environ.copy()
    env["JAVA_HOME"] = JDK_DIR
    sig = subprocess.run([APKSIGNER, "verify", "--verbose", APK_PATH], capture_output=True, text=True, env=env, check=True)
    for line in sig.stdout.splitlines():
        if "Verified" in line or "Verifies" in line:
            print(f"  {line}")

    print("\nALL APK INTEGRITY & SIGNATURE CHECKS PASSED!")


if __name__ == "__main__":
    main()

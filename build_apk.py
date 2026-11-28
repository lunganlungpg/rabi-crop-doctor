import os
import shutil
import subprocess
import zipfile
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP1_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
WEB_DIR = os.path.join(BASE_DIR, "web")
PROJECT_DIR = os.path.join(BASE_DIR, "android_build")
SRC_DIR = os.path.join(PROJECT_DIR, "src")
RES_DIR = os.path.join(PROJECT_DIR, "res")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
MANIFEST = os.path.join(PROJECT_DIR, "AndroidManifest.xml")
BUILD_DIR = os.path.join(PROJECT_DIR, "build")
GEN_DIR = os.path.join(BUILD_DIR, "gen")
OBJ_DIR = os.path.join(BUILD_DIR, "obj")
DEX_DIR = os.path.join(BUILD_DIR, "dex")

def _resolve_jdk_dir():
    candidates = [
        os.environ.get("JAVA_HOME", ""),
        os.path.join(APP1_DIR, "jdk-17", "jdk-17.0.12+7"),
        r"C:\Users\lungh\.p2\pool\plugins\org.eclipse.justj.openjdk.hotspot.jre.full.win32.x86_64_21.0.10.v20260205-0638\jre",
    ]
    for cand in candidates:
        if cand and os.path.exists(os.path.join(cand, "bin", "javac.exe")):
            return cand
    javac = shutil.which("javac")
    if javac:
        return os.path.dirname(os.path.dirname(javac))
    return candidates[1]


JDK_DIR = _resolve_jdk_dir()
JDK_BIN = os.path.join(JDK_DIR, "bin")
SDK_DIR = r"C:\Users\lungh\AppData\Local\Android\Sdk"
BUILD_TOOLS = os.path.join(SDK_DIR, "build-tools", "35.0.1")
ANDROID_JAR = os.path.join(SDK_DIR, "platforms", "android-35", "android.jar")

JAVAC = os.path.join(JDK_BIN, "javac.exe")
KEYTOOL = os.path.join(JDK_BIN, "keytool.exe")
AAPT2 = os.path.join(BUILD_TOOLS, "aapt2.exe")
D8 = os.path.join(BUILD_TOOLS, "d8.bat")
ZIPALIGN = os.path.join(BUILD_TOOLS, "zipalign.exe")
APKSIGNER = os.path.join(BUILD_TOOLS, "apksigner.bat")

KEYSTORE_PATH = os.path.join(PROJECT_DIR, "debug.keystore")
FINAL_APK = os.path.join(BASE_DIR, "RabiCropDoctor.apk")


def run(cmd, desc, env=None):
    print(f"\n---> {desc} ...")
    res = subprocess.run(cmd, shell=False, capture_output=True, text=True, env=env)
    if res.stdout:
        print(f"[STDOUT]: {res.stdout.strip()}")
    if res.stderr:
        print(f"[STDERR]: {res.stderr.strip()}")
    if res.returncode != 0:
        raise RuntimeError(f"Step '{desc}' failed with exit code {res.returncode}")
    print(f"SUCCESS: {desc}")


def generate_pwa_icons():
    for sz in (192, 512):
        out = os.path.join(WEB_DIR, f"icon_{sz}.png")
        img = Image.new("RGBA", (sz, sz), (20, 83, 45, 255))
        draw = ImageDraw.Draw(img)
        pad = int(sz * 0.14)
        # Leaf diamond-oval body
        draw.pieslice([pad, pad, sz - pad, sz - pad], 210, 390, fill=(34, 197, 94, 255))
        draw.ellipse([int(sz * 0.28), int(sz * 0.24), int(sz * 0.72), int(sz * 0.76)], fill=(34, 197, 94, 255))
        # Midrib vein
        draw.line([(int(sz * 0.34), int(sz * 0.72)), (int(sz * 0.68), int(sz * 0.28))], fill=(236, 253, 245, 255), width=max(4, sz // 36))
        # Amber corner scanner frame
        b_w = max(5, sz // 32)
        c_l = int(sz * 0.18)
        draw.rectangle([pad, pad, sz - pad, sz - pad], outline=(251, 191, 36, 255), width=b_w)
        # Re-cover center edges so only corner brackets appear
        draw.rectangle([pad + c_l, pad - 1, sz - pad - c_l, pad + b_w + 1], fill=(20, 83, 45, 255))
        draw.rectangle([pad + c_l, sz - pad - b_w - 1, sz - pad - c_l, sz - pad + 1], fill=(20, 83, 45, 255))
        img.save(out)


def sync_web_assets():
    if os.path.exists(ASSETS_DIR):
        shutil.rmtree(ASSETS_DIR)
    shutil.copytree(WEB_DIR, ASSETS_DIR)
    print(f"Synced web/ assets -> {ASSETS_DIR}")


def main():
    generate_pwa_icons()
    sync_web_assets()

    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    for d in (GEN_DIR, OBJ_DIR, DEX_DIR):
        os.makedirs(d, exist_ok=True)

    # 1. Compile Android resources
    compiled_res = os.path.join(BUILD_DIR, "compiled_res.zip")
    run([AAPT2, "compile", "--dir", RES_DIR, "-o", compiled_res], "Compile Android resources with AAPT2")

    # 2. Link resources & assets
    base_apk = os.path.join(BUILD_DIR, "base.apk")
    run(
        [
            AAPT2,
            "link",
            "-I",
            ANDROID_JAR,
            "--manifest",
            MANIFEST,
            "-o",
            base_apk,
            "--java",
            GEN_DIR,
            compiled_res,
            "-A",
            ASSETS_DIR,
            "--auto-add-overlay",
        ],
        "Link resources, manifest, and offline AI assets",
    )

    # 3. Compile Java sources
    java_files = []
    for root_p in (SRC_DIR, GEN_DIR):
        for r, _, files in os.walk(root_p):
            for f in files:
                if f.endswith(".java"):
                    java_files.append(os.path.join(r, f))

    run([JAVAC, "--release", "11", "-cp", ANDROID_JAR, "-d", OBJ_DIR] + java_files, "Compile Java classes")

    # 4. Convert bytecode to DEX with D8
    class_files = []
    for r, _, files in os.walk(OBJ_DIR):
        for f in files:
            if f.endswith(".class"):
                class_files.append(os.path.join(r, f))

    env = os.environ.copy()
    env["JAVA_HOME"] = JDK_DIR
    env["PATH"] = JDK_BIN + ";" + env.get("PATH", "")

    run([D8, "--min-api", "21", "--lib", ANDROID_JAR, "--output", DEX_DIR] + class_files, "Compile DEX bytecode with D8", env=env)

    # 5. Add classes.dex into base.apk
    dex_file = os.path.join(DEX_DIR, "classes.dex")
    with zipfile.ZipFile(base_apk, "a") as apk_zip:
        apk_zip.write(dex_file, "classes.dex")

    # 6. Align APK
    aligned_apk = os.path.join(BUILD_DIR, "aligned.apk")
    run([ZIPALIGN, "-v", "-p", "4", base_apk, aligned_apk], "Align APK pages with zipalign")

    # 7. Ensure signing keystore exists
    if not os.path.exists(KEYSTORE_PATH):
        run(
            [
                KEYTOOL,
                "-genkeypair",
                "-v",
                "-keystore",
                KEYSTORE_PATH,
                "-storepass",
                "android",
                "-alias",
                "androiddebugkey",
                "-keypass",
                "android",
                "-keyalg",
                "RSA",
                "-keysize",
                "2048",
                "-validity",
                "10000",
                "-dname",
                "CN=Rabi Crop Doctor,O=ICAR-IASRI Community,C=IN",
            ],
            "Generate APK signing keystore",
        )

    # 8. Sign APK
    if os.path.exists(FINAL_APK):
        os.remove(FINAL_APK)
    run(
        [
            APKSIGNER,
            "sign",
            "--ks",
            KEYSTORE_PATH,
            "--ks-pass",
            "pass:android",
            "--ks-key-alias",
            "androiddebugkey",
            "--key-pass",
            "pass:android",
            "--out",
            FINAL_APK,
            aligned_apk,
        ],
        "Sign APK with apksigner",
        env=env,
    )

    # 9. Verify Signature
    run([APKSIGNER, "verify", "--verbose", FINAL_APK], "Verify cryptographic APK signature", env=env)

    size_mb = os.path.getsize(FINAL_APK) / 1024.0 / 1024.0
    print("\n" + "=" * 62)
    print(f"APK BUILD SUCCESSFUL: {FINAL_APK} ({size_mb:.2f} MB)")
    print("=" * 62)


if __name__ == "__main__":
    main()

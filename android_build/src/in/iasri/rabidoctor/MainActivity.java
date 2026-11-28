package in.iasri.rabidoctor;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.print.PrintAttributes;
import android.print.PrintDocumentAdapter;
import android.print.PrintManager;
import android.provider.MediaStore;
import android.util.Base64;
import android.view.View;
import android.webkit.JavascriptInterface;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ProgressBar;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends Activity {

    private static final int REQ_CAMERA_PERM = 1001;
    private static final int REQ_CAMERA_CAPTURE = 1002;
    private static final int REQ_GALLERY_PICK = 1003;
    private static final int REQ_FILE_CHOOSER = 1004;

    private WebView webView;
    private ProgressBar progressBar;
    private ValueCallback<Uri[]> filePathCallback;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        webView = findViewById(R.id.webview);
        progressBar = findViewById(R.id.progressBar);

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setUseWideViewPort(true);
        settings.setLoadWithOverviewMode(true);
        settings.setSupportZoom(false);

        webView.setLayerType(View.LAYER_TYPE_HARDWARE, null);
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);

        webView.addJavascriptInterface(new WebAppInterface(this), "AndroidBridge");

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onProgressChanged(WebView view, int newProgress) {
                if (newProgress < 100) {
                    progressBar.setVisibility(View.VISIBLE);
                    progressBar.setProgress(newProgress);
                } else {
                    progressBar.setVisibility(View.GONE);
                }
            }

            @Override
            public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (filePathCallback != null) {
                    filePathCallback.onReceiveValue(null);
                }
                filePathCallback = callback;
                try {
                    Intent galleryIntent = new Intent(Intent.ACTION_GET_CONTENT);
                    galleryIntent.addCategory(Intent.CATEGORY_OPENABLE);
                    galleryIntent.setType("image/*");
                    startActivityForResult(Intent.createChooser(galleryIntent, "Select Leaf Image"), REQ_FILE_CHOOSER);
                    return true;
                } catch (Exception e) {
                    filePathCallback = null;
                    return false;
                }
            }
        });

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, String url) {
                if (url.startsWith("file:///android_asset/")) {
                    return false;
                }
                try {
                    startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(url)));
                    return true;
                } catch (Exception e) {
                    return false;
                }
            }
        });

        webView.loadUrl("file:///android_asset/index.html");
    }

    private void launchCameraInternal() {
        try {
            File outFile = CameraFileProvider.getCaptureFile(this);
            if (outFile.exists()) {
                outFile.delete();
            }
            Uri photoUri = CameraFileProvider.getCaptureUri();
            Intent camIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            camIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
            camIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            startActivityForResult(camIntent, REQ_CAMERA_CAPTURE);
        } catch (Exception e) {
            Toast.makeText(this, "Camera launch failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void launchGalleryInternal() {
        try {
            Intent pickIntent = new Intent(Intent.ACTION_GET_CONTENT);
            pickIntent.addCategory(Intent.CATEGORY_OPENABLE);
            pickIntent.setType("image/*");
            startActivityForResult(Intent.createChooser(pickIntent, "Choose Leaf Photo"), REQ_GALLERY_PICK);
        } catch (Exception e) {
            Toast.makeText(this, "Gallery unavailable: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_CAMERA_PERM) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                launchCameraInternal();
            } else {
                Toast.makeText(this, "Camera permission denied — please use Gallery photo picker.", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQ_FILE_CHOOSER) {
            if (filePathCallback != null) {
                Uri[] results = null;
                if (resultCode == RESULT_OK && data != null && data.getData() != null) {
                    results = new Uri[]{data.getData()};
                }
                filePathCallback.onReceiveValue(results);
                filePathCallback = null;
            }
            return;
        }

        if (resultCode != RESULT_OK) {
            return;
        }

        if (requestCode == REQ_CAMERA_CAPTURE) {
            File file = CameraFileProvider.getCaptureFile(this);
            if (file.exists() && file.length() > 0) {
                dispatchResizedBitmapToJs(Uri.fromFile(file), file.getAbsolutePath());
            } else if (data != null && data.getExtras() != null && data.getExtras().get("data") instanceof Bitmap) {
                sendBitmapToJs((Bitmap) data.getExtras().get("data"));
            }
        } else if (requestCode == REQ_GALLERY_PICK && data != null && data.getData() != null) {
            dispatchResizedBitmapToJs(data.getData(), null);
        }
    }

    private void dispatchResizedBitmapToJs(final Uri uri, final String filePath) {
        new Thread(() -> {
            try {
                BitmapFactory.Options bounds = new BitmapFactory.Options();
                bounds.inJustDecodeBounds = true;
                try (InputStream is = getContentResolver().openInputStream(uri)) {
                    BitmapFactory.decodeStream(is, null, bounds);
                }

                int maxDim = Math.max(bounds.outWidth, bounds.outHeight);
                int sample = 1;
                while (maxDim / sample > 512) {
                    sample *= 2;
                }

                BitmapFactory.Options opts = new BitmapFactory.Options();
                opts.inSampleSize = sample;
                Bitmap bmp;
                try (InputStream is = getContentResolver().openInputStream(uri)) {
                    bmp = BitmapFactory.decodeStream(is, null, opts);
                }
                if (bmp == null) return;

                int rot = readExifRotation(uri, filePath);
                if (rot != 0) {
                    Matrix m = new Matrix();
                    m.postRotate(rot);
                    Bitmap rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), m, true);
                    if (rotated != bmp) bmp.recycle();
                    bmp = rotated;
                }

                sendBitmapToJs(bmp);
            } catch (Exception e) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Error reading photo: " + e.getMessage(), Toast.LENGTH_SHORT).show());
            }
        }).start();
    }

    private int readExifRotation(Uri uri, String filePath) {
        try {
            ExifInterface exif = null;
            if (filePath != null) {
                exif = new ExifInterface(filePath);
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                try (InputStream is = getContentResolver().openInputStream(uri)) {
                    if (is != null) exif = new ExifInterface(is);
                }
            }
            if (exif == null) return 0;
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            if (orientation == ExifInterface.ORIENTATION_ROTATE_90) return 90;
            if (orientation == ExifInterface.ORIENTATION_ROTATE_180) return 180;
            if (orientation == ExifInterface.ORIENTATION_ROTATE_270) return 270;
        } catch (Exception ignored) {
        }
        return 0;
    }

    private void sendBitmapToJs(Bitmap bmp) {
        int w = bmp.getWidth();
        int h = bmp.getHeight();
        float scale = Math.min(1.0f, 768.0f / Math.max(w, h));
        Bitmap finalBmp = bmp;
        if (scale < 1.0f) {
            finalBmp = Bitmap.createScaledBitmap(bmp, Math.round(w * scale), Math.round(h * scale), true);
            if (finalBmp != bmp) bmp.recycle();
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        finalBmp.compress(Bitmap.CompressFormat.JPEG, 90, baos);
        finalBmp.recycle();

        final String dataUrl = "data:image/jpeg;base64," + Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP);
        runOnUiThread(() -> webView.evaluateJavascript("if(window.onNativeImageSelected){window.onNativeImageSelected('" + dataUrl + "');}", null));
    }

    public class WebAppInterface {
        Context mContext;

        WebAppInterface(Context c) {
            mContext = c;
        }

        @JavascriptInterface
        public void capturePhoto() {
            runOnUiThread(() -> {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M &&
                        checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA_PERM);
                } else {
                    launchCameraInternal();
                }
            });
        }

        @JavascriptInterface
        public void pickGalleryPhoto() {
            runOnUiThread(MainActivity.this::launchGalleryInternal);
        }

        @JavascriptInterface
        public void shareText(String title, String content) {
            runOnUiThread(() -> {
                try {
                    Intent sendIntent = new Intent(Intent.ACTION_SEND);
                    sendIntent.putExtra(Intent.EXTRA_TITLE, title);
                    sendIntent.putExtra(Intent.EXTRA_TEXT, content);
                    sendIntent.setType("text/plain");
                    startActivity(Intent.createChooser(sendIntent, title));
                } catch (Exception e) {
                    Toast.makeText(mContext, "Share error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                }
            });
        }

        @JavascriptInterface
        public void copyText(String text) {
            runOnUiThread(() -> {
                try {
                    ClipboardManager clipboard = (ClipboardManager) mContext.getSystemService(Context.CLIPBOARD_SERVICE);
                    clipboard.setPrimaryClip(ClipData.newPlainText("Rabi Crop Report", text));
                    Toast.makeText(mContext, "Copied!", Toast.LENGTH_SHORT).show();
                } catch (Exception e) {
                    Toast.makeText(mContext, "Copy failed", Toast.LENGTH_SHORT).show();
                }
            });
        }

        @JavascriptInterface
        public void showToast(String message) {
            runOnUiThread(() -> Toast.makeText(mContext, message, Toast.LENGTH_SHORT).show());
        }

        @JavascriptInterface
        public void saveFile(String filename, String content, String mimeType) {
            runOnUiThread(() -> {
                try {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        ContentValues values = new ContentValues();
                        values.put(MediaStore.MediaColumns.DISPLAY_NAME, filename);
                        values.put(MediaStore.MediaColumns.MIME_TYPE, mimeType);
                        values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);
                        Uri uri = mContext.getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
                        if (uri != null) {
                            try (OutputStream os = mContext.getContentResolver().openOutputStream(uri)) {
                                if (os != null) {
                                    os.write(content.getBytes("UTF-8"));
                                    Toast.makeText(mContext, "Saved to Downloads: " + filename, Toast.LENGTH_LONG).show();
                                    return;
                                }
                            }
                        }
                    }
                    File downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
                    if (!downloadsDir.exists()) downloadsDir.mkdirs();
                    File file = new File(downloadsDir, filename);
                    try (FileOutputStream fos = new FileOutputStream(file)) {
                        fos.write(content.getBytes("UTF-8"));
                    }
                    Toast.makeText(mContext, "Saved: " + file.getAbsolutePath(), Toast.LENGTH_LONG).show();
                } catch (Exception e) {
                    Toast.makeText(mContext, "Save failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
                }
            });
        }

        @JavascriptInterface
        public void printDocument() {
            runOnUiThread(() -> {
                try {
                    PrintManager printManager = (PrintManager) getSystemService(Context.PRINT_SERVICE);
                    PrintDocumentAdapter printAdapter = webView.createPrintDocumentAdapter("Rabi_Field_Incidence_Report");
                    printManager.print("Rabi Field Incidence Report", printAdapter, new PrintAttributes.Builder().build());
                } catch (Exception e) {
                    Toast.makeText(mContext, "Print error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                }
            });
        }
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }
}

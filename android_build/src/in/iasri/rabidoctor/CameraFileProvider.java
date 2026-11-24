package in.iasri.rabidoctor;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.FileNotFoundException;

/**
 * Dependency-free ContentProvider exposing cache capture JPEGs cleanly for
 * ACTION_IMAGE_CAPTURE across Android 7-15 without external AndroidX libraries.
 */
public class CameraFileProvider extends ContentProvider {

    public static final String AUTHORITY = "in.iasri.rabidoctor.fileprovider";
    private static final String CAPTURE_NAME = "leaf_capture.jpg";

    public static Uri getCaptureUri() {
        return Uri.parse("content://" + AUTHORITY + "/" + CAPTURE_NAME);
    }

    public static File getCaptureFile(android.content.Context ctx) {
        return new File(ctx.getCacheDir(), CAPTURE_NAME);
    }

    @Override
    public boolean onCreate() {
        return true;
    }

    @Override
    public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        File file = getCaptureFile(getContext());
        if (file.getParentFile() != null && !file.getParentFile().exists()) {
            file.getParentFile().mkdirs();
        }
        int pfdMode = ParcelFileDescriptor.parseMode(mode);
        return ParcelFileDescriptor.open(file, pfdMode);
    }

    @Override
    public Cursor query(Uri uri, String[] projection, String selection, String[] selectionArgs, String sortOrder) {
        File file = getCaptureFile(getContext());
        if (projection == null) {
            projection = new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE};
        }
        MatrixCursor cursor = new MatrixCursor(projection);
        Object[] row = new Object[projection.length];
        for (int i = 0; i < projection.length; i++) {
            if (OpenableColumns.DISPLAY_NAME.equals(projection[i])) {
                row[i] = CAPTURE_NAME;
            } else if (OpenableColumns.SIZE.equals(projection[i])) {
                row[i] = file.length();
            }
        }
        cursor.addRow(row);
        return cursor;
    }

    @Override
    public String getType(Uri uri) {
        return "image/jpeg";
    }

    @Override
    public Uri insert(Uri uri, ContentValues values) {
        return null;
    }

    @Override
    public int delete(Uri uri, String selection, String[] selectionArgs) {
        return 0;
    }

    @Override
    public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
        return 0;
    }
}

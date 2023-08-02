package edu.ubp.djlai;

import java.awt.image.BufferedImage;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author agustin
 */
class Mat2Image {

    static {
        OpenCV.loadLocally();
    }
    Mat mat = new Mat();
    BufferedImage img;
    byte[] dat;

    public Mat2Image() {
    }

    public Mat2Image(Mat mat) {
        getSpace(mat);
    }

    public void getSpace(Mat mat) {
        this.mat = mat;
        int width = mat.width();
        int height = mat.height();
        int type = mat.channels() != 1 ? BufferedImage.TYPE_3BYTE_BGR : BufferedImage.TYPE_BYTE_GRAY;
        if (type == BufferedImage.TYPE_3BYTE_BGR) {
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);
        }
        dat = new byte[width * height * (int) mat.elemSize()];
        img = new BufferedImage(width, height, type);
    }

    BufferedImage getImage(Mat mat) {
        getSpace(mat);
        mat.get(0, 0, dat);
        img.getRaster().setDataElements(0, 0,
                mat.cols(), mat.rows(), dat);
        return img;
    }

}

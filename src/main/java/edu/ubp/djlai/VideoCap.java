package edu.ubp.djlai;

import java.awt.image.BufferedImage;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

/**
 *
 * @author agustin
 */
public class VideoCap {
    static{
        OpenCV.loadLocally();
    }

    VideoCapture cap;
    Mat2Image mat2Img = new Mat2Image();
    DjlAi djl;

    VideoCap(DjlAi djl){
        this.djl = djl;
        this.cap = new VideoCapture();
        this.cap.open(0);
    } 
 
    BufferedImage getOneFrame() {
        this.cap.read(this.mat2Img.mat);
        Imgproc.resize(this.mat2Img.mat, this.mat2Img.mat, new Size(320, 240), 1.0, 1.0, Imgproc.INTER_AREA);
        Object objImg = this.djl.detectar(this.mat2Img.mat);        
        return mat2Img.getImage((Mat) objImg);
    }
}

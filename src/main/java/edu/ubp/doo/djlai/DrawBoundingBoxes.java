package edu.ubp.doo.djlai;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Rect;

/**
 *
 * @author agustin
 */
public class DrawBoundingBoxes {

    public static void draw(List<Classifications.Classification> objs, Image img) throws IOException {
        Mat matrix = (Mat) img.getWrappedImage();
        //Debo redimensionar la imagen con openCV
        //para poder ver los recuadros con las detecciones
        Imgproc.resize(matrix, matrix, new Size(640, 640));
        Mat subMatrix = new Mat();
        for (Classifications.Classification obj : objs) {
            String detectedClassName = obj.getClassName();
            double probability = obj.getProbability();//;
            DecimalFormat df = new DecimalFormat("0.00");
//            String textToAdd = detectedClassName + " (" + df.format(probability) + ")";
            String textToAdd = " " + df.format(probability) + " ";
            
            DetectedObjects.DetectedObject objConvered = (DetectedObjects.DetectedObject) obj;

            BoundingBox box = objConvered.getBoundingBox();
            Rectangle rec = box.getBounds();   
            Rect rectCrop = new Rect(new double[]{
                rec.getX(),
                rec.getY(),
                rec.getWidth(),
                rec.getHeight()
            });
            subMatrix = matrix.submat(rectCrop);
            Imgcodecs.imwrite("/home/agustin/Escritorio/UBP/Investigacion/patente.jpg", subMatrix); // to save an image
            Imgproc.rectangle(matrix, // Matrix obj of the image
                    new Point(rec.getX(), rec.getY()), // p1
                    new Point(rec.getX() + rec.getWidth(), rec.getY() + rec.getHeight()), // p2
                    new Scalar(0, 255, 25), // Scalar object for color BGR
                    1 // Thickness of the line
            );

            int baseline[] = {0};
            Size sizeText = Imgproc.getTextSize(textToAdd, Imgproc.FONT_ITALIC, 0.3, 1, baseline);
            Imgproc.rectangle(matrix, // Matrix obj of the image
                    new Point(rec.getX(), rec.getY()), // p1
                    new Point((rec.getX() + sizeText.width + 3), (rec.getY() + sizeText.height + 5)), // p2
                    new Scalar(0, 255, 25), // Scalar object for color BGR
                    -1 // Thickness of the line. If -1 fill the rectangle
            );
            Imgproc.putText(matrix,
                    textToAdd,
                    new Point(rec.getX() + 2, rec.getY() + 10),
                    Imgproc.FONT_ITALIC,
                    0.35,
                    new Scalar(0, 0, 255),
                    1);
        }
        Imgcodecs.imwrite("/home/agustin/Escritorio/UBP/Investigacion/res.jpg", matrix); // to save an image
        // display Image
        HighGui.imshow("Result", matrix);
        // Waiting for a key event to delay
        HighGui.waitKey();
        System.exit(0);
    }

    public static void draw(DetectedObjects detecteds, Image img) throws IOException {
        Mat matrix = (Mat) img.getWrappedImage();
        String res = detecteds.toJson();

        System.out.println(res);
        String jsonString = res;
        JSONArray jsonArr = new JSONArray(jsonString);
//        List<BoundingBox> boxes = new ArrayList<>();
//        List<String> names = new ArrayList<>();
//        List<Double> prob = new ArrayList<>();
        //Debo redimensionar la imagen con openCV
        //para poder ver los recuadros con las detecciones
        Imgproc.resize(matrix, matrix, new Size(640, 640));
        Mat subMatrix = new Mat();
        for (int i = 0; i < jsonArr.length(); i++) {
            JSONObject jsonObj = jsonArr.getJSONObject(i);

            String detectedClassName = jsonObj.getString("className");
            double probability = jsonObj.getDouble("probability") * 100;
            DecimalFormat df = new DecimalFormat("0");
            String textToAdd = detectedClassName + " (" + df.format(probability) + "%)";

            JSONObject firstPoint = jsonObj.getJSONObject("boundingBox").getJSONArray("corners").getJSONObject(0);
            double x1 = firstPoint.getDouble("x");
            double y1 = firstPoint.getDouble("y");

            JSONObject secondPoint = jsonObj.getJSONObject("boundingBox").getJSONArray("corners").getJSONObject(2);
            double x2 = secondPoint.getDouble("x");
            double y2 = secondPoint.getDouble("y");
            Rectangle rec = new Rectangle(x1, y1, x2, y2);
//            boxes.add(rec);
//            names.add(detectedClassName);
//            prob.add(probability);
            //            patenteImg = img.getSubImage((int)rec.getX(), (int)rec.getY(), (int)rec.getWidth(), (int)rec.getHeight());
            Rect rectCrop = new Rect(new double[]{
                rec.getX(),
                rec.getY(),
                rec.getWidth(),
                rec.getHeight()
            });
            subMatrix = matrix.submat(rectCrop);
            Imgcodecs.imwrite("/home/agustin/Escritorio/UBP/Investigacion/patente.jpg", subMatrix); // to save an image
            Imgproc.rectangle(matrix, // Matrix obj of the image
                    new Point(x1, y1), // p1
                    new Point(x2, y2), // p2
                    new Scalar(0, 255, 25), // Scalar object for color BGR
                    1 // Thickness of the line
            );

            int baseline[] = {0};
            Size sizeText = Imgproc.getTextSize(textToAdd, Imgproc.FONT_ITALIC, 0.5, 1, baseline);
            Imgproc.rectangle(matrix, // Matrix obj of the image
                    new Point(x1, y1), // p1
                    new Point((x1 + sizeText.width + 5), (y1 + sizeText.height + 5)), // p2
                    new Scalar(0, 255, 25), // Scalar object for color BGR
                    -1 // Thickness of the line. If -1 for fill the rectangle
            );
            Imgproc.putText(matrix,
                    textToAdd,
                    new Point(x1 + 2, y1 + 12),
                    Imgproc.FONT_ITALIC,
                    0.5,
                    new Scalar(0, 0, 255),
                    1);
        }
//        img = img.resize(640, 640, true);
//        DetectedObjects converted = new DetectedObjects(names, prob, boxes);
//        saveBoundingBoxImage(img, converted);
        Imgcodecs.imwrite("/home/agustin/Escritorio/UBP/Investigacion/res.jpg", matrix); // to save an image
        // display Image
        HighGui.imshow("Result", matrix);
        // Waiting for a key event to delay
        HighGui.waitKey();

    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("/home/agustin/Escritorio/UBP/Investigacion/");
        Files.createDirectories(outputDir);

        img.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("detected.png");
        // OpenJDK can't save jpg with alpha channel
        img.save(Files.newOutputStream(imagePath), "png");
    }
}

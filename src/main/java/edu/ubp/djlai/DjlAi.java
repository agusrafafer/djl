package edu.ubp.djlai;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import java.awt.image.BufferedImage;
import java.io.IOException;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author agustin
 */
public class DjlAi {

    static {
        OpenCV.loadLocally();
    }

    ZooModel<Image, DetectedObjects> model;
    Predictor<Image, DetectedObjects> predictor;
    ImageFactory factory;

    public DjlAi() {
        this.model = loadModel();
        this.predictor = model.newPredictor();
        this.factory = ImageFactory.getInstance();
    }

    public Object detectar(Mat frame) {
        try {
//        Mat reduced = new Mat();
//        Imgproc.resize(frame, reduced, new Size(320, 240), 1.0, 1.0, Imgproc.INTER_AREA);
//        frame = reduced;
//            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);
            Image img = factory.fromImage(frame);
            long start = System.currentTimeMillis();
            DetectedObjects detections = predictor.predict(img);
            long end = System.currentTimeMillis();
            System.out.println("Elapsed time: " + (end - start) + " ms");
            img.drawBoundingBoxes(detections);
            return img.getWrappedImage();
        } catch (Exception e) {
            return null;
        }
    }

//    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException, ModelException, InterruptedException {
////        var img = ImageFactory.getInstance().fromUrl("https://resources.djl.ai/images/dog_bike_car.jpg");
////        img.getWrappedImage();
//        ZooModel<Image, DetectedObjects> model = loadModel();
//        Predictor<Image, DetectedObjects> predictor = model.newPredictor();
//
//        OpenCV.loadLocally();
////        var criteria = Criteria.builder()
////                .setTypes(Image.class, DetectedObjects.class)
////                .optArtifactId("ssd")
////                .optProgress(new ProgressBar())
////                .build();
////        var model = criteria.loadModel();
////        var predictor = model.newPredictor();
//        Mat frame = new Mat();
//        VideoCapture cap = new VideoCapture();// Load video using the videocapture method//
//        cap.open(0);//"rtsp://localhost:8554/mystream");
//        if (!cap.isOpened()) {
//            System.out.println("No se detecto la camara");
//            return;
//        }
//
//        boolean captured = false;
//        for (int i = 0; i < 10; ++i) {
//            captured = cap.read(frame);
//            if (captured) {
//                break;
//            }
//            try {
//                Thread.sleep(50);
//            } catch (InterruptedException ignore) {
//                // ignore
//            }
//        }
//        if (!captured) {
//            JOptionPane.showConfirmDialog(null, "Fallo la captura de la imagen desde la camara");
//        }
//
//        JFrame jframe = new JFrame("Video"); // the lines below create a frame to display the resultant video with object detection and localization//
//        JLabel vidpanel = new JLabel();
//        jframe.setContentPane(vidpanel);
//        jframe.setSize(600, 600);
//        jframe.setVisible(true);// we instantiate the frame here//
//        jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        ImageFactory factory = ImageFactory.getInstance();
//
//        while (cap.isOpened()) {
//            if (!cap.read(frame)) {
//                break;
//            }
//            Mat reduced = new Mat();
//            Imgproc.resize(frame, reduced, new Size(320, 240), 1.0, 1.0, Imgproc.INTER_AREA);
//            frame = reduced;
////            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);
//            Image img = factory.fromImage(frame);
////                MatOfByte mob = new MatOfByte();
////                Imgcodecs.imencode(".jpg", frame, mob);
////                byte ba[] = mob.toArray();
////                var img = ImageFactory.getInstance().fromInputStream(new ByteArrayInputStream(ba));
////                img.getWrappedImage();
//            long start = System.currentTimeMillis();
//            DetectedObjects detections = predictor.predict(img);
//            long end = System.currentTimeMillis();
//            System.out.println("Elapsed time: " + (end - start) + " ms");
//            img.drawBoundingBoxes(detections);
////                img.getWrappedImage();
////                Path outputDir = Paths.get("/home/agustin/");
////                Path imagePath = outputDir.resolve("detected-dog_bike_car.png");
////                img.save(Files.newOutputStream(imagePath), "png"); 
//            ImageIcon icon = new ImageIcon(toBufferedImage((Mat) img.getWrappedImage()));
//            vidpanel.setIcon(icon);
//            vidpanel.repaint();
//
//        }
//
////        ImageIO.write(img, "png", new File("/home/agustin/ssd.png"));
//        model.close();
//    }
    
    private ZooModel<Image, DetectedObjects> loadModel() {
        try {
            Criteria<Image, DetectedObjects> criteria
                    = Criteria.builder()
                            .optApplication(Application.CV.OBJECT_DETECTION)
//                            .optModelUrls("/home/agustin/best_01jul2022_3.onnx")
                            .setTypes(Image.class, DetectedObjects.class)
                            .optEngine("OnnxRuntime")
                            .optProgress(new ProgressBar())
                            .build();
            //YOLOv5

        for (ModelZoo object : BasicModelZoo.listModelZoo()) {
            System.out.println(object.toString());
        }
            return criteria.loadModel();
        } catch (MalformedModelException | ModelNotFoundException | IOException e) {
            return null;
        }
    }

    private static BufferedImage toBufferedImage(Mat mat) {
        int width = mat.width();
        int height = mat.height();
        int type = mat.channels() != 1 ? BufferedImage.TYPE_3BYTE_BGR : BufferedImage.TYPE_BYTE_GRAY;

        if (type == BufferedImage.TYPE_3BYTE_BGR) {
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);
        }

        byte[] data = new byte[width * height * (int) mat.elemSize()];
        mat.get(0, 0, data);

        BufferedImage ret = new BufferedImage(width, height, type);
        ret.getRaster().setDataElements(0, 0, width, height, data);

        return ret;
    }
}

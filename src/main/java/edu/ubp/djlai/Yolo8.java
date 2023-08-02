package edu.ubp.djlai;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import nu.pattern.OpenCV;

/**
 *
 * @author agustin
 */
public class Yolo8 {

    public static final String PATH_IMAGEN = "/home/agustin/Imágenes/auto.png";
    public static final String PATH_MODELO = "file:///home/agustin/Escritorio/UBP/Investigacion/Modelos-YoloV8-patentes/modelo_patentes_yolov8_13jul2023.onnx";
    
    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        System.setProperty("org.jooq.no-logo", "true");
        OpenCV.loadLocally();
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(640, 640));
        pipeline.add(new ToTensor());

        Translator<Image, DetectedObjects> translator = new Yolo8Translator(640, 640);

        Criteria<Image, DetectedObjects> criteria
                = Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optModelUrls(PATH_MODELO)
                        .optTranslator(translator)
                        .optProgress(new ProgressBar())
                        .optEngine("OnnxRuntime")
                        .build();

        ZooModel<Image, DetectedObjects> model = criteria.loadModel();

        Image img = ImageFactory.getInstance().fromFile(Paths.get(PATH_IMAGEN));
        Predictor<Image, DetectedObjects> predictor = model.newPredictor();
        long tInicio = System.nanoTime();
        DetectedObjects patentes_detectadas = predictor.predict(img);
        long tFin = System.nanoTime();
        float tiempo = ((tFin - tInicio) / 1000000);
        System.out.println("Tiempo predicción: " + tiempo + " m seg");
        //topK me quedo con la mejor predicción
        tInicio = System.nanoTime();
        List<Classifications.Classification> objs = patentes_detectadas.topK(1);
        System.out.println(patentes_detectadas);
        DrawBoundingBoxes.draw(objs, img);
        tFin = System.nanoTime();
        tiempo = ((tFin - tInicio) / 1000000);
        System.out.println("Tiempo postProcesado: " + tiempo + " m seg");
    }
}

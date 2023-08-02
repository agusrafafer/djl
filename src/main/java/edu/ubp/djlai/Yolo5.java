package edu.ubp.djlai;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import nu.pattern.OpenCV;

/**
 *
 * @author agustin
 */
public class Yolo5 {

    public static final String PATH_IMAGEN = "/home/agustin/Imágenes/auto.png";
    public static final String PATH_MODELO = "file:///home/agustin/Escritorio/UBP/Investigacion/Modelos-YoloV5-patentes/best_01jul2022_3.onnx";
    public static final String PATH_SINSET = "file:///home/agustin/NetBeansProjects/djlAi/src/main/resources/synset.txt";

    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        System.setProperty("org.jooq.no-logo", "true");
        OpenCV.loadLocally();
        Pipeline pipeline = new Pipeline();
        pipeline.add(new Resize(640, 640));
        pipeline.add(new ToTensor());

        Translator<Image, DetectedObjects> translator = YoloV5Translator
                .builder()
                .setPipeline(pipeline)
                .optSynsetUrl(PATH_SINSET)
                .optThreshold(0.1f)
                .build();

        Criteria<Image, DetectedObjects> criteria
                = Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optDevice(Device.cpu())
                        .optModelUrls(PATH_MODELO)
                        .optTranslator(translator)
                        .optProgress(new ProgressBar())
                        .optEngine("OnnxRuntime")
                        .build();

        for (ModelZoo modeloZoo : ModelZoo.listModelZoo()) {
            System.out.println(modeloZoo);
        }
        ZooModel<Image, DetectedObjects> model = criteria.loadModel();

        Image img = ImageFactory.getInstance().fromFile(Paths.get(PATH_IMAGEN));
        Predictor<Image, DetectedObjects> predictor = model.newPredictor();
        long tInicio = System.nanoTime();
        DetectedObjects objDetectados = predictor.predict(img);
        long tFin = System.nanoTime();
        float tiempo = (tFin - tInicio) / 1000000;
        System.out.println("Tiempo predicción: " + tiempo);
        //topK me quedo con la mejor predicción
        List<Classifications.Classification> objs = objDetectados.topK(1);
        System.out.println(objDetectados);
        DrawBoundingBoxes.draw(objs, img);
    }
}

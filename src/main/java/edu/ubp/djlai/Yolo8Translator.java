package edu.ubp.djlai;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author agustin
 */
public class Yolo8Translator implements Translator<Image, DetectedObjects> {

    private final int imageWidth;
    private final int imageHeight;

    public Yolo8Translator(int imageWidth, int imageHeight) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
        NDArray array = input.toNDArray(ctx.getNDManager());
        Pipeline pipeline = new Pipeline()
                .add(new Resize(this.imageWidth, this.imageHeight))
                .add(new ToTensor());
        return pipeline.transform(new NDList(array));

    }

    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList ndlist) throws Exception {
        NDArray ndarray = ndlist.get(0);
//        Shape[] sh = ndlist.getShapes();
//        long rows = sh[0].get(1);
//        long dimensions = sh[0].get(0);
        Path pathSinset = Paths.get("/home/agustin/NetBeansProjects/djlAi/src/main/resources/synset.txt");
        List<String> listClasses = new ArrayList<>(Files.readAllLines(pathSinset));
        List<BoundingBox> listBoundingBox = new ArrayList<>();
        List<Double> listConfidences = new ArrayList<>();
        List<Long> listClassId = new ArrayList<>();
        //Obtengo la mayor predicción
        NDArray mayor = ndarray.get(new NDIndex("4:84,:")).max();
        //Obtengo la posicion en el eje "y" de la mayor predicción. La matriz output es de 84x8400
        long posMayor = ndarray.get(new NDIndex("4:84,:")).argMax(1).getLong();
        float maxClassScore = mayor.getFloat();
        listConfidences.add(Float.valueOf(maxClassScore).doubleValue());
        //Me quedo con las preimeras 4 posiciones para construir el boundingBox 
        float[] arr = ndarray.get(new NDIndex("0:4," + posMayor)).toFloatArray();
        
        //Pensar (Ver) como obtener el id correcto de las clases
        long classId = mayor.argMax().getLong();
        listClassId.add(classId);
        float x = arr[0];
        float y = arr[1];
        float w = arr[2];
        float h = arr[3];
        float xLeft = (x - (w / 2));
        float yTop = (y - (h / 2));
        listBoundingBox.add(new Rectangle(xLeft, yTop, w, h));

//     De aquí para abajo este código es ineficiente
//        for (long i = (rows - 1); i >= 0; i--) {
//            intermedio = ndarray.get(new NDIndex("4:84," + i));
//            //Para saber la posicion de la clase que posee el mayor score
//            float maxClassScore = intermedio.max().getFloat();
//            if (maxClassScore > 0.8f) {
//                float[] arrProbs = intermedio.toFloatArray();
//                float[] arrPtos = ndarray.get(new NDIndex("0:4," + i)).toFloatArray();
//                long classId = intermedio.argMax().getLong();
//                listClassId.add(classId);
//                listConfidences.add(Float.valueOf(maxClassScore).doubleValue());
//
//                //Falta ubicar correctamente la caja 
//                //para que no ocurra un problema al
//                //tratar de cortar la misma con opencv
//                float x = arrPtos[0];
//                float y = arrPtos[1];
//                float w = arrPtos[2];
//                float h = arrPtos[3];
//                float xLeft = (x - (w / 2));
//                float yTop = (y - (h / 2));
//                listBoundingBox.add(new Rectangle(xLeft, yTop, w, h));
//                //break;
//            }
//        }

        return new DetectedObjects(listClasses, listConfidences, listBoundingBox);
    }

}

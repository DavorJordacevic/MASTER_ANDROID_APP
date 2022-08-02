package com.fincore.master.encoder

import android.util.Log
import org.opencv.core.*
import org.opencv.core.Core.norm
import org.opencv.core.Core.transpose
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net

// This is a singleton class (it can have only one instance) user for preprocessing, encoding and comparing the faces
object EncoderProcessor{
    // Default number of faces used for comparison
    private val TARGET_IMG_WIDTH = 112.0
    private val TARGET_IMG_HEIGHT = 112.0
    private val MEAN = Scalar(0.485, 0.456, 0.406)
    lateinit var dnnNet: Net
    private var layerNames = mutableListOf<String>()

    init {
        println("Singleton encoder class invoked.")
    }

    // Load the ONNX model and update the network backend
    // TODO Test VULKAN performance on Snapdragon processors
    fun updateBackend(path: String){
        Log.d("ENCODER", "INITIALIZATION")
        dnnNet = Dnn.readNetFromONNX(path)
        dnnNet.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV)
        dnnNet.setPreferableTarget(Dnn.DNN_TARGET_CPU)
        // Set the default layer names and keypoints
        setDefaultValues()
    }

    // Set the default layer names and keypoints
    fun setDefaultValues(){
        layerNames = dnnNet.unconnectedOutLayersNames
    }

    // Preprocess faces
    fun preprocess(img: Mat): Mat {
        // // Create an input blob
        val blob = Dnn.blobFromImage(
            img,
            1.0,
            Size(
                TARGET_IMG_WIDTH,
                TARGET_IMG_HEIGHT
            ),
            MEAN,
            true,
            false
        )

        return blob
    }

    // Predict the encodings
    fun predict(img: Mat): Mat {
            // read and process the input face images
            val inputBlob = preprocess(img)
            dnnNet.setInput(inputBlob)
            // inference
            val result = mutableListOf<Mat>()
            dnnNet.forward(result, layerNames)
            val transposed = Mat()
            transpose(result[0], transposed)
        return transposed
    }

    // Compare faces using the cosine distance
    fun compare(vectorA: Mat, vectorB: Mat): Double {
        val cosDist = 1.0 - vectorA.dot(vectorB) / (norm(vectorA) * norm(vectorB))
        Log.d("cosDistEncoder", cosDist.toString())
        return cosDist
    }

}

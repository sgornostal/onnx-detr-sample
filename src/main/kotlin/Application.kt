import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtUtil
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Graphics2D
import java.awt.Image
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.File
import java.nio.FloatBuffer
import java.nio.file.Paths
import javax.imageio.ImageIO
import kotlin.math.exp

/**
 * @author Slava Gornostal
 */
fun main() {
    val imagePath = Paths.get("test.jpg")
    val image = ImageIO.read(imagePath.toFile())

    val modelPath = "detr.onnx"
    val env = OrtEnvironment.getEnvironment()
    val session = env.createSession(modelPath)

    val imageBuffer = preprocessImage(image)
    val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imageBuffer), longArrayOf(1, 3, 800, 800))
    val inputMap = mapOf(session.inputNames.first() to inputTensor)
    val outputs = session.run(inputMap)
    val logits = (outputs[0] as OnnxTensor).floatBuffer.array()
    val boxes = (outputs[1] as OnnxTensor).floatBuffer.array()
    val result = postProcessOutput(intArrayOf(image.width, image.height), logits, boxes)
    val graphics = image.createGraphics() as Graphics2D
    val padding = 5
    graphics.font = graphics.font.deriveFont(13f)
    graphics.color = Color.WHITE
    graphics.stroke = BasicStroke(1.7f)
    for (obj in result) {
        graphics.drawRect(obj.x.toInt(), obj.y.toInt(), (obj.width - obj.x).toInt(), (obj.height - obj.y).toInt())
        graphics.drawString(obj.className, obj.x.toInt(), (obj.y - padding).toInt())
    }
    graphics.dispose()
    ImageIO.write(image, "jpg", File("result.jpg"))
}

data class DetectedObject(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val className: String,
    val score: Float
)

// https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
// IN [1-5]

private fun preprocessImage(image: BufferedImage): FloatArray {
    // Resize the image
    val resizedImage = BufferedImage(800, 800, BufferedImage.TYPE_3BYTE_BGR)
    val graphics = resizedImage.createGraphics()
    graphics.drawImage(image.getScaledInstance(800, 800, Image.SCALE_SMOOTH), 0, 0, null)
    graphics.dispose()
    // Extract bytes from the image
    val pixels = (resizedImage.raster.dataBuffer as DataBufferByte).data
    val floatArray = FloatArray(3 * 800 * 800)
    val imageMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    val imageStd = floatArrayOf(0.229f, 0.224f, 0.225f)
    // Normalize and rearrange the dimensions
    for (y in 0..<800) {
        for (x in 0..<800) {
            val pixelIndex = y * 800 * 3 + x * 3
            // Convert byte value to float in [0,1]
            val blue = (pixels[pixelIndex].toInt() and 0xFF) / 255.0f
            val green = (pixels[pixelIndex + 1].toInt() and 0xFF) / 255.0f
            val red = (pixels[pixelIndex + 2].toInt() and 0xFF) / 255.0f
            // The channels need to be in (C, H, W) order
            floatArray[y * 800 + x] = (red - imageMean[0]) / imageStd[0]
            floatArray[800 * 800 + y * 800 + x] = (green - imageMean[1]) / imageStd[1]
            floatArray[2 * 800 * 800 + y * 800 + x] = (blue - imageMean[2]) / imageStd[2]
        }
    }
    return floatArray
}

private fun postProcessOutput(
    inputSize: IntArray,
    logitsArray: FloatArray,
    boxesArray: FloatArray,
    threshold: Float = 0.8f
): List<DetectedObject> {
    // COCO dataset
    val numClasses = 92L
    // det defaul
    val numQueries = 100L
    val logits = OrtUtil.reshape(logitsArray, longArrayOf(numQueries, numClasses)) as Array<FloatArray>
    val probas = softmax2D(logits)
    val detectedObjects = mutableListOf<DetectedObject>()
    // skip N/A
    for (classId in 0..<numClasses) {
        if (classId == 0L) continue
        // Extract the probabilities of the class of interest (excluding the last one)
        val classProbas = probas.map { it[classId.toInt()] }.toFloatArray()
        // Find indices where probability is above threshold
        val keepIndices = classProbas.withIndex().filter { it.value > threshold }.map { it.index }
        val boxes = OrtUtil.reshape(boxesArray, longArrayOf(numQueries, 4)) as Array<FloatArray>
        // Filter probabilities and boxes
        val probasToKeep = keepIndices.map { probas[it] }.toTypedArray()
        val boxesToKeep = keepIndices.map { boxes[it] }.toTypedArray()
        for (i in probasToKeep.indices) {
            val x = (boxesToKeep[i][0])
            val y = boxesToKeep[i][1]
            val width = boxesToKeep[i][2]
            val height = boxesToKeep[i][3]
            detectedObjects.add(
                DetectedObject(
                    x = (x - 0.5f * width) * (inputSize[0]),
                    y = (y - 0.5f * height) * (inputSize[1]),
                    width = (x + 0.5f * width) * (inputSize[0]),
                    height = (y + 0.5f * height) * (inputSize[1]),
                    className = CLASSES[classId.toInt()],
                    score = probasToKeep[i][classId.toInt()]
                )
            )
        }
    }
    return detectedObjects
}

private fun softmax2D(x: Array<FloatArray>): Array<FloatArray> {
    return x.map { row ->
        val max = row.maxOrNull() ?: 0f  // Subtracting the max improves numerical stability
        val eX = row.map { exp((it - max).toDouble()).toFloat() }.toFloatArray()
        val sum = eX.sum()
        eX.map { it / sum }.toFloatArray()
    }.toTypedArray()
}

val CLASSES = arrayOf(
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush", "N/A"
)
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using System.Threading.Tasks;
using PassthroughCameraSamples;

//LOOK AT https://github.com/Unity-Technologies/inference-engine-samples/blob/main/BlazeDetectionSample/Face/Assets/Scripts/BlazeUtils.cs
// SampleImageAffine to do image sampling on the GPU and store in a tensor
/* or blit?
using UnityEngine;

public class WebCamToRenderTexture : MonoBehaviour
{
    public WebCamTexture webCamTexture; // Assign in the inspector
    public RenderTexture renderTexture; // Assign in the inspector

    void Start()
    {
        if (webCamTexture == null)
        {
            webCamTexture = new WebCamTexture();
            if (!webCamTexture.isPlaying)
            {
                webCamTexture.Play();
            }
        }


    }

    void Update()
    {
        if (webCamTexture != null && renderTexture != null)
        {
            //Blit the WebCamTexture to the RenderTexture
            Graphics.Blit(webCamTexture, renderTexture);
        }
    }
}
*/

public class RunYOLO8nPose : MonoBehaviour
{
    [Header("Detection Settings")]
    [SerializeField] private WebCamTextureManager webCamTextureManager;
    [SerializeField] private ObjectRenderer objectRenderer;
    private WebCamTexture _webcamTexture;

    // Drag the yolov8_pose.onnx file here
    public ModelAsset asset;
    private Worker engine;
    private BackendType backend = BackendType.GPUPixel;
    private Texture2D _cpuTexture;

    private const int numJoints = 17;
    private const int maxPeople = 1;    

    //Image size for the model
    private const int imageWidth = 640;
    private const int imageHeight = 640;

    [SerializeField, Range(0, 1)] float iouThreshold = 0.5f;
    [SerializeField, Range(0, 1)] float scoreThreshold = 0.5f;

    Tensor centersToCorners;

    Tensor<float> webcamTextureTensor;
    RenderTexture renderTexture;
    public struct Keypoint
    {
        public float x;
        public float y;
        public float confidence;
    }       
    //bounding box data
    public struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
        public string label;
    }

    public void Start()
    {
        Debug.Log("[ObjectDetector] Starting up and acquiring webcam texture.");
        _webcamTexture = webCamTextureManager.WebCamTexture;
        if (_webcamTexture != null)
        {
            _cpuTexture = new Texture2D(_webcamTexture.width, _webcamTexture.height, TextureFormat.RGBA32, false);
            Debug.Log($"[ObjectDetector] WebCamTexture dimensions: {_webcamTexture.width}x{_webcamTexture.height}");
            //webcamTextureTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));
            renderTexture = new RenderTexture(_webcamTexture.width, _webcamTexture.height, 0, RenderTextureFormat.ARGB32);
            renderTexture.enableRandomWrite = true;
            renderTexture.Create();
        }
        else
        {
            Debug.LogError("[ObjectDetector] WebCamTexture is null at Start.");
        }

        LoadModel(backend);
    }
  
   void LoadModel(BackendType backend){
        var model1 = ModelLoader.Load(asset); 
        var centersToCornersData = new[]
            {
                        1,      0,      1,      0,
                        0,      1,      0,      1,
                        -0.5f,  0,      0.5f,   0,
                        0,      -0.5f,  0,      0.5f
            };

        var graph = new FunctionalGraph();
        var input = graph.AddInput(model1, 0);
        var modelOutput = Functional.Forward(model1, input)[0];

        var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);
        var scores = modelOutput[0, 4, ..];
        var keypointsData = modelOutput[0, 5.., ..].Transpose(0,1);

        var boxCorners = Functional.MatMul(boxCoords, Functional.Constant(new TensorShape(4, 4), centersToCornersData));

        var indices = Functional.NMS(boxCorners, scores, iouThreshold, scoreThreshold);

        var indicesExpandedBox = indices.Unsqueeze(-1).BroadcastTo(new[] {4});
        var indicesExpandedKpts = indices.Unsqueeze(-1).BroadcastTo(new[] {51});
        
        var finalCoords = Functional.Gather(boxCoords, 0, indicesExpandedBox);
        var finalKeypointsData = Functional.Gather(keypointsData, 0, indicesExpandedKpts);

        var model2 = graph.Compile(finalCoords, finalKeypointsData);

        //Create engine to run model
        engine = new Worker(model2, backend);
    }

    async void Update()
    {
        Debug.Log("[ObjectDetector] Update.");
        if (isProcessing){
            Debug.Log("[ObjectDetector] isProcessing == true... exiting Update().");
            return;
        }
        if (!_webcamTexture)
        {
            _webcamTexture = webCamTextureManager.WebCamTexture;

            if (_webcamTexture)
            {
                //_cpuTexture = new Texture2D(_webcamTexture.width, _webcamTexture.height, TextureFormat.RGBA32, false);
                renderTexture = new RenderTexture(_webcamTexture.width, _webcamTexture.height, 0, RenderTextureFormat.ARGB32);  // Matching TextureFormat.RGBA32 used by _cpuTexture
                renderTexture.enableRandomWrite = true;
                renderTexture.Create();
                Debug.Log("[ObjectDetector] WebCamTexture is now available; texture created.");
            }
        }

        if(renderTexture == null)
        {
            Debug.Log("[ObjectDetector] RenderTexture is null.");
            return;
        }
        if(!renderTexture.IsCreated())
        {
            Debug.LogError("[ObjectDetector] RenderTexture is not created.");
            return;
        }
        Debug.Log("[ObjectDetector] blitting webcam texture to render texture.");
        Graphics.Blit(_webcamTexture, renderTexture);
        //_cpuTexture.SetPixels(_webcamTexture.GetPixels());
        //_cpuTexture.Apply();        
        
        //await ExecuteModel(_cpuTexture);
        await ExecuteModel(renderTexture);
    }

    bool isProcessing = false;

    async Task ExecuteModel(RenderTexture inputTexture)
    {
        Debug.Log("[ObjectDetector] Executing model.");
        if (inputTexture == null)
        {
            Debug.Log("[ObjectDetector] Input texture is null.");
            return;
        }
        if (isProcessing)
        {
            Debug.Log("[ObjectDetector] isProcessing == true... exiting ExecuteModel().");
            return;
        }

        isProcessing = true;

        //var inputTensor = TextureConverter.ToTensor(inputTexture, imageWidth, imageHeight, 3);
        //engine.Schedule(inputTensor);
        TextureTransform settings = new TextureTransform().SetDimensions(imageWidth, imageHeight, 3).SetTensorLayout(TensorLayout.NHWC);

        if(webcamTextureTensor == null)
        {
            webcamTextureTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));
        }

        TextureConverter.RenderToTexture(webcamTextureTensor, inputTexture, settings);

        // YOLO Has a a layer NonMaxSuppression that needs to be run on the CPU.  This is a limitation of the current Unity Inference Engine.
        // So yolo will be sending data to and from the CPU which KILLS frames per second.
        engine.Schedule(webcamTextureTensor);

        using var output_ = engine.PeekOutput(0) as Tensor<float>;
        using var ketPoints_ = engine.PeekOutput(1) as Tensor<float>;

        using var outputTensor = await output_.ReadbackAndCloneAsync();
        using var keyPointsTensor = await ketPoints_.ReadbackAndCloneAsync();

        objectRenderer.RenderPoseDetections(outputTensor, keyPointsTensor);

        isProcessing = false;
    }

    private void OnDestroy()
    {
        centersToCorners?.Dispose();
        engine?.Dispose();

        if (_cpuTexture != null)
        {
            Destroy(_cpuTexture);
            _cpuTexture = null;
        }        
    }
}
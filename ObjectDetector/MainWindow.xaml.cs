using Emgu.CV;
using Emgu.CV.Structure;
using Microsoft.ML;
using ObjectDetector.Extensions;
using ObjectDetector.Models;
using OnnxObjectDetection;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Rectangle = System.Windows.Shapes.Rectangle;
using Window = System.Windows.Window;
using Timer = System.Timers.Timer;
using Point = System.Drawing.Point;

namespace ObjectDetector
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private VideoCapture? capture;
        private CancellationTokenSource? cameraCaptureCancellationTokenSource;

        private OnnxOutputParser? outputParser;
        private PredictionEngine<ImageInputData, TinyYoloPrediction>? tinyYoloPredictionEngine;
        private PredictionEngine<ImageInputData, CustomVisionPrediction>? customVisionPredictionEngine;

        private WriteableBitmap? _writableBitmap;
        private Dictionary<BoundingBox, ObjectTrackInfo> _objectTracking = new Dictionary<BoundingBox, ObjectTrackInfo>();
        private Timer _frameStartTimer = new Timer(5000) { AutoReset = true,Enabled = true};//5seconds
        private bool _canDetectObjects = true;
        private float _detectionTreshold = 0.45f;
        private float _nmsTreshold = 0.35f;
        private bool _isCustomModel = true;

        private static readonly string modelsDirectory = Path.Combine(Environment.CurrentDirectory, @"Assets\OnnxModels");

        public MainWindow()
        {
            InitializeComponent();
            _frameStartTimer.Elapsed += (s, e) => _canDetectObjects = true;
            _isCustomModel = LoadModel(true);
            this.Loaded += MainWindow_Loaded;
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            _writableBitmap = new WriteableBitmap(Convert.ToInt32(ActualWidth), Convert.ToInt32(ActualHeight), 96, 96, PixelFormats.Bgr24, null);
            WebCamImage.Source = _writableBitmap;
        }

        protected override void OnActivated(EventArgs e)
        {
            base.OnActivated(e);
            StartCameraCapture();
        }

        protected override void OnDeactivated(EventArgs e)
        {
            base.OnDeactivated(e);
            StopCameraCapture();
        }

        private bool LoadModel(bool tinyModel)
        {
            // Check for an Onnx model exported from Custom Vision
            var customVisionExport = Directory.GetFiles(modelsDirectory, "*.zip").FirstOrDefault();

            // If there is one, use it.
            if (customVisionExport is not null && !tinyModel)
            {
                var customVisionModel = new CustomVisionModel(customVisionExport);
                var modelConfigurator = new OnnxModelConfigurator(customVisionModel);

                outputParser = new OnnxOutputParser(customVisionModel);
                customVisionPredictionEngine = modelConfigurator.GetMlNetPredictionEngine<CustomVisionPrediction>();
                return true;
            }
            else // Otherwise default to Tiny Yolo Onnx model
            {
                var tinyYoloModel = new TinyYoloModel(Path.Combine(modelsDirectory, "TinyYolo2_model.onnx"));
                var modelConfigurator = new OnnxModelConfigurator(tinyYoloModel);

                outputParser = new OnnxOutputParser(tinyYoloModel);
                tinyYoloPredictionEngine = modelConfigurator.GetMlNetPredictionEngine<TinyYoloPrediction>();
                return false;
            }
        }

        private void StartCameraCapture()
        {
            cameraCaptureCancellationTokenSource = new CancellationTokenSource();
            Task.Run(() => CaptureCamera(cameraCaptureCancellationTokenSource.Token), cameraCaptureCancellationTokenSource.Token);
        }

        private void StopCameraCapture() => cameraCaptureCancellationTokenSource?.Cancel();

        private async Task CaptureCamera(CancellationToken token)
        {
            if (capture is null)
                capture = new VideoCapture();

            if (capture is not null)
            {
                capture.Start();
                if (capture.IsOpened)
                {
                    _frameStartTimer.Start();
                    while (!token.IsCancellationRequested)
                    {
                        var image = capture.QueryFrame().ToImage<Bgr, byte>();
                        await Application.Current.Dispatcher.InvokeAsync(() =>
                        {
                            _writableBitmap?.WritePixels(new Int32Rect(0, 0, image.Width, image.Height), image.MIplImage.ImageData, image.MIplImage.ImageSize, image.MIplImage.WidthStep);
                        });
                        await ParseWebCamFrame(image, token);
                    }
                    capture.Stop();
                    capture.Dispose();
                }
            }
        }
       
        async Task ParseWebCamFrame(Image<Bgr,Byte> image, CancellationToken token)
        {
            if (customVisionPredictionEngine == null && tinyYoloPredictionEngine == null)
                return;

            var filteredBoxes = (_canDetectObjects, _isCustomModel, new ImageInputData { Image = image.ToBitmap() }) switch
            {
                (true, true, ImageInputData input) => input.DetectObjectsUsingModel(customVisionPredictionEngine!, outputParser!, 5, _detectionTreshold),
                (true, false, ImageInputData input) => input.DetectObjectsUsingModel(tinyYoloPredictionEngine!, outputParser!, 5, _detectionTreshold),
                (_, _, _) => new List<BoundingBox>()
            };

            _canDetectObjects = false;//reset to false to avoid object detection at every single frame and let the timer reactivate detection


            _objectTracking = image.Mat.TrackObjects(filteredBoxes, _objectTracking)
                                        .RemoveDuplicates(_detectionTreshold, _nmsTreshold);

            _objectTracking.Values.Where(x => x.MaxDisappearance <= 0).Select(x => OnObjectDesappearing(x));

            if (!token.IsCancellationRequested)
            {
                await Application.Current.Dispatcher.InvokeAsync(() =>
                {
                    DrawOverlays(_objectTracking, WebCamImage.ActualHeight, WebCamImage.ActualWidth);
                });
            }
        }

        

        private void DrawOverlays(List<BoundingBox> filteredBoxes, double originalHeight, double originalWidth)
        {
            WebCamCanvas.Children.Clear();

            foreach (var box in filteredBoxes)
            {
                // process output boxes
                double x = Math.Max(box.Dimensions.X, 0);
                double y = Math.Max(box.Dimensions.Y, 0);
                double width = Math.Min(originalWidth - x, box.Dimensions.Width);
                double height = Math.Min(originalHeight - y, box.Dimensions.Height);

                // fit to current image size
                x = originalWidth * x / ImageSettings.imageWidth;
                y = originalHeight * y / ImageSettings.imageHeight;
                width = originalWidth * width / ImageSettings.imageWidth;
                height = originalHeight * height / ImageSettings.imageHeight;

                var boxColor = box.BoxColor.ToMediaColor();

                var objBox = new Rectangle
                {
                    Width = width,
                    Height = height,
                    Fill = new SolidColorBrush(Colors.Transparent),
                    Stroke = new SolidColorBrush(boxColor),
                    StrokeThickness = 2.0,
                    Margin = new Thickness(x, y, 0, 0)
                };

                var objDescription = new TextBlock
                {
                    Margin = new Thickness(x + 4, y + 4, 0, 0),
                    Text = box.Description,
                    FontWeight = FontWeights.Bold,
                    Width = 126,
                    Height = 21,
                    TextAlignment = TextAlignment.Center
                };

                var objDescriptionBackground = new Rectangle
                {
                    Width = 134,
                    Height = 29,
                    Fill = new SolidColorBrush(boxColor),
                    Margin = new Thickness(x, y, 0, 0)
                };

                WebCamCanvas.Children.Add(objDescriptionBackground);
                WebCamCanvas.Children.Add(objDescription);
                WebCamCanvas.Children.Add(objBox);
            }
        }

        private void DrawOverlays(Dictionary<BoundingBox, ObjectTrackInfo> objects, double originalHeight, double originalWidth)
        {
            WebCamCanvas.Children.Clear();

            foreach (var item in objects)
            {
                // process output boxes
                double x = Math.Max(item.Value.CurrentBox.X, 0);
                double y = Math.Max(item.Value.CurrentBox.Y, 0);
                double width = Math.Min(originalWidth - x, item.Value.CurrentBox.Width);
                double height = Math.Min(originalHeight - y,item.Value.CurrentBox.Height);

                // fit to current image size
                x = originalWidth * x / ImageSettings.imageWidth;
                y = originalHeight * y / ImageSettings.imageHeight;
                width = originalWidth * width / ImageSettings.imageWidth;
                height = originalHeight * height / ImageSettings.imageHeight;

                var boxColor = item.Value.InitialBoundingBox.BoxColor.ToMediaColor();

                var objBox = new Rectangle
                {
                    Width = width,
                    Height = height,
                    Fill = new SolidColorBrush(Colors.Transparent),
                    Stroke = new SolidColorBrush(boxColor),
                    StrokeThickness = 2.0,
                    Margin = new Thickness(x, y, 0, 0)
                };

                var position = item.Value.CurrentBox.Location.X < 0 ? "Out" : "In";

                var objDescription = new TextBlock
                {
                    Margin = new Thickness(x + 4, y + 4, 0, 0),
                    Text = $"{item.Value.InitialBoundingBox.Description}({position})",
                    FontWeight = FontWeights.Bold,
                    Width = 146,
                    Height = 21,
                    TextAlignment = TextAlignment.Center
                };

                var objDescriptionBackground = new Rectangle
                {
                    Width = 150,
                    Height = 29,
                    Fill = new SolidColorBrush(boxColor),
                    Margin = new Thickness(x, y, 0, 0)
                };

                WebCamCanvas.Children.Add(objDescriptionBackground);
                WebCamCanvas.Children.Add(objDescription);
                WebCamCanvas.Children.Add(objBox);
            }
        }


        private int OnObjectDesappearing(ObjectTrackInfo objectTrackInfo)
        {
            MessageBox.Show($"{objectTrackInfo.InitialBoundingBox.Description} has disappeared");
            return 0;
        }


    }

    internal static class ColorExtensions
    {
        internal static System.Windows.Media.Color ToMediaColor(this System.Drawing.Color drawingColor)
        {
            return System.Windows.Media.Color.FromArgb(drawingColor.A, drawingColor.R, drawingColor.G, drawingColor.B);
        }
    }
}

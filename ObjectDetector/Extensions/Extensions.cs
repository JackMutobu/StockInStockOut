using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Microsoft.ML;
using ObjectDetector.Models;
using OnnxObjectDetection;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace ObjectDetector.Extensions
{
    public static class Extensions
    {
        public static Rectangle ToRectangle(this BoundingBox box)
            => new Rectangle((int)box.Dimensions.X, (int)box.Dimensions.Y, (int)box.Dimensions.Width, (int)box.Dimensions.Height);

        public static ObjectTrackInfo InitializeTracking(this BoundingBox box, Mat image)
        {
            var tracker = new TrackerCSRT();
            var currentBox = box.ToRectangle();
            tracker.Init(image, currentBox);
            return new ObjectTrackInfo(box, tracker, currentBox);
        }

        public static ObjectTrackInfo UpdateTracking(this ObjectTrackInfo objectTrackInfo, Mat image)
        {
            var isFound = objectTrackInfo.Tracker.Update(image, out Rectangle currentBox);

            return objectTrackInfo.Update(currentBox, hasDesapeared:!isFound);
        }

        public static Point GetCenter(int width, int height) => new Point(width / 2, height / 2);

        public static Point GetCenter(this (int Width, int Height) dimensions) => new Point(dimensions.Width / 2, dimensions.Height / 2);

        public static Point GetCenter(this (double Width, double Height) dimensions) => new Point(Convert.ToInt32(dimensions.Width) / 2, Convert.ToInt32(dimensions.Height) / 2);

        public static List<BoundingBox> DetectObjectsUsingModel(this ImageInputData imageInputData, PredictionEngine<ImageInputData,TinyYoloPrediction> predictionEngine, OnnxOutputParser outputParser, int limit, float treshold)
        {

            var labels = predictionEngine.Predict(imageInputData).PredictedLabels;
            var boundingBoxes = outputParser?.ParseOutputs(labels);
            var filteredBoxes = outputParser?.FilterBoundingBoxes(boundingBoxes, limit, treshold);

            return filteredBoxes ?? new List<BoundingBox>();
        }

        public static List<BoundingBox> DetectObjectsUsingModel(this ImageInputData imageInputData, PredictionEngine<ImageInputData, CustomVisionPrediction> predictionEngine, OnnxOutputParser outputParser, int limit, float treshold)
        {

            var labels = predictionEngine.Predict(imageInputData).PredictedLabels;
            var boundingBoxes = outputParser?.ParseOutputs(labels);
            var filteredBoxes = outputParser?.FilterBoundingBoxes(boundingBoxes, limit, treshold);

            return filteredBoxes ?? new List<BoundingBox>();
        }

        public static Dictionary<BoundingBox, ObjectTrackInfo> TrackObjects(this Mat image, List<BoundingBox> filteredBoxes, Dictionary<BoundingBox, ObjectTrackInfo> objects)
        {
            var newBoxes = filteredBoxes.Except(objects.Keys);
            var newTrackedObjects = newBoxes.Select(x => x.InitializeTracking(image));
            var updatedTrackedObjects = objects.Values.Select(x => x.UpdateTracking(image));

            return newTrackedObjects.Concat(updatedTrackedObjects).ToDictionary(x => x.InitialBoundingBox);

        }

        public static Dictionary<BoundingBox, ObjectTrackInfo> RemoveDuplicates(this Dictionary<BoundingBox, ObjectTrackInfo> objects, float detectionTreshold,float nmsTreshold)
        {
            var values = objects.Values.ToList();
            var bboxes = objects.Values.Select(x => x.CurrentBox).ToArray();
            var scores = objects.Values.Select(x => x.InitialBoundingBox.Confidence).ToArray();
            var resultIndices = DnnInvoke.NMSBoxes(bboxes, scores, detectionTreshold, nmsTreshold);

            return resultIndices.Select(i => values[i]).ToDictionary(x => x.InitialBoundingBox, x => x);
        }
    }
}

using Emgu.CV;
using OnnxObjectDetection;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace ObjectDetector.Models
{
    public record ObjectTrackInfo(BoundingBox InitialBoundingBox, TrackerCSRT Tracker, Rectangle CurrentBox,bool HasDesapeared = false, bool IsInitialized = true, int MaxDisappearance = 10)
    {
        public List<Rectangle> MotionHistory { get; init; } = new List<Rectangle>();

        public ObjectTrackInfo Update(Rectangle currentBox,bool hasDesapeared =  false, bool isInitialized = true)
        {
            var maxDisappearance = MaxDisappearance;
            if (!hasDesapeared)
                MotionHistory.Add(CurrentBox);
            else
                maxDisappearance = maxDisappearance - 1;


            return new ObjectTrackInfo(InitialBoundingBox,Tracker, currentBox,hasDesapeared, isInitialized,MaxDisappearance: maxDisappearance)
            {
                MotionHistory = MotionHistory
            };
        }
    }
}

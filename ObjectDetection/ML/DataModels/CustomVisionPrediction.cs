using Microsoft.ML.Data;

namespace OnnxObjectDetection
{
    public class CustomVisionPrediction : OnnxObjectPrediction
    {
        [ColumnName("model_outputs0")]
        public new float[] PredictedLabels { get; set; }
    }
}
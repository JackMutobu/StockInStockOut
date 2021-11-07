using Microsoft.ML.Data;

namespace OnnxObjectDetection
{
    public class TinyYoloPrediction : OnnxObjectPrediction
    {
        [ColumnName("grid")]
        public new float[] PredictedLabels { get; set; }
    }
}
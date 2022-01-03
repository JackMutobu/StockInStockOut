using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ObjectDetect.Shared
{
    public class ProductDetection
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;

        public bool? IsIn { get; set; }

        public DateTime Date { get; set; }
    }
    public interface IObjectDetectClient
    {
        Task OnDetection(ProductDetection product);
    }
}

using Microsoft.AspNetCore.SignalR;
using ObjectDetect.Shared;

namespace ObjectDetectServer.Hub
{
    public class ObjectDetectHub: Hub<IObjectDetectClient>
    {
        public async Task OnDetection(ProductDetection product)
        {
            await Clients.All.OnDetection(product);
        }
    }
}

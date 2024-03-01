using System;
using System.ClientModel;
using System.ClientModel.Primitives;

namespace Azure.Pinecone;

public class AzurePineconeClient
{
    ClientPipeline _pipeline;
    Uri _uri;

    public AzurePineconeClient(Uri uri, ApiKeyCredential apiKey)
    {
        _pipeline = ClientPipeline.Create();
    }

    public void Upsert()
    {
        PipelineMessage message = _pipeline.CreateMessage();
        PipelineRequest request = message.Request;
        request.Uri = new Uri(_uri.ToString() + "/vectors/upsert");
    }
}
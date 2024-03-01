using OpenAI.Embeddings;
using Pinecone;
using Pinecone.Grpc;
using System;
using System.Collections.Generic;

namespace OpenAI.Tests.Chat;

public partial class AzureExtensionsTests
{
    public class PineconeVectorbaseStore : VectorbaseStore
    {
        PineconeClient _client;
        string _indexName;
        Index<GrpcTransport> _index;
        int _nextId = -1;

        public PineconeVectorbaseStore(PineconeClient client, string indexName)
        {
            _indexName = indexName;
            _client = client;
        }

        public override int Add(VectorbaseEntry entry)
        {
            EnsureIndex();

            Vector vector = CreateVector(entry, _nextId);
            _index.Upsert([vector]);
            return _nextId++;
        }

        private void EnsureIndex()
        {
            if (_index == null)
            {
                _index = _client.GetIndex(_indexName).GetAwaiter().GetResult();
                _nextId = 0;
            }
        }

        public override void Add(IReadOnlyList<VectorbaseEntry> entries)
        {
            EnsureIndex();

            List<Vector> vectors = new List<Vector>(entries.Count);
            foreach (var entry in entries)
            {
                Vector vector = CreateVector(entry, _nextId++);
                vectors.Add(vector);
            }
            _index.Upsert(vectors);
        }

        public override IEnumerable<VectorbaseEntry> Find(ReadOnlyMemory<float> vector, FindOptions options)
        {
            EnsureIndex();

            ScoredVector[] vectors = _index.Query(vector.ToArray(), (uint)options.MaxEntries, includeMetadata: true).GetAwaiter().GetResult();
            var result = new List<VectorbaseEntry>(options.MaxEntries);   
            foreach(var scoredVector in vectors)
            {
                if (scoredVector.Score>options.Threshold)
                {
                    var id = int.Parse(scoredVector.Id);
                    MetadataValue data = scoredVector.Metadata["data"];
                    BinaryData bd = BinaryData.FromString((string)data.Inner);
                    VectorbaseEntry entry = new VectorbaseEntry(ReadOnlyMemory<float>.Empty, bd, id);
                    result.Add(entry);
                }
            }
            return result;
        }

        private static Vector CreateVector(VectorbaseEntry entry, int id)
        {
            MetadataMap metadata = new();
            MetadataValue value = entry.Data.ToString();
            metadata.Add("data", value);
            Vector vector = new()
            {
                Values = entry.Vector.ToArray(),
                Id = id.ToString(),
                Metadata = metadata
            };
            return vector;
        }
    }
}
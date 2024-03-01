// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using OpenAI.Embeddings;
using System.Collections.Generic;
using System;

namespace OpenAI.Embeddings;

public class Vectorbase
{
    readonly EmbeddingClient _client;
    readonly VectorbaseStore _store;

    readonly List<string> _todo = new List<string>();

    public Vectorbase(EmbeddingClient client, VectorbaseStore store = default)
    {
        _client = client;
        _store = store??new MemoryVectorbaseStore();
    }

    public void Add(string text)
    {
        lock (_todo) {
            _todo.Add(text);
        }
    }   

    public IEnumerable<VectorbaseEntry> Find(string text, FindOptions options = default)
    {
        if (options == default) options = new FindOptions();
        if (_todo.Count>0) ProcessToDo();
        ReadOnlyMemory<float> textVector = GetEmbedding(text);
        IEnumerable<VectorbaseEntry> nearest = _store.Find(textVector, options);
        return nearest;
    }

    private void ProcessToDo()
    {
        lock (_todo)
        {
            EmbeddingCollection embeddings = _client.GenerateEmbeddings(_todo);

            foreach (Embedding embedding in embeddings)
            {
                ReadOnlyMemory<float> vector = embedding.Vector;
                string item = _todo[(int)embedding.Index];
                _store.Add(new VectorbaseEntry(vector, BinaryData.FromString(item)));
            }
            _todo.Clear();
        }
    }

    private ReadOnlyMemory<float> GetEmbedding(string fact)
    {
        Embedding embedding = _client.GenerateEmbedding(fact);
        return embedding.Vector;
    }
}

public class FindOptions
{
    public int MaxEntries { get; set; } = 3;
    public float Threshold { get; set; } = 0.25f;
}




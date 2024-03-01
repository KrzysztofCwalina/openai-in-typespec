// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Generic;
using System;

namespace OpenAI.Embeddings;

public abstract class VectorbaseStore
{
    public abstract IEnumerable<VectorbaseEntry> Find(ReadOnlyMemory<float> vector, FindOptions options);

    public abstract int Add(VectorbaseEntry entry);

    public abstract void Add(IReadOnlyList<VectorbaseEntry> entry);

    public static float CosineSimilarity(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        float dot = 0, xSumSquared = 0, ySumSquared = 0;

        for (int i = 0; i < x.Length; i++)
        {
            dot += x[i] * y[i];
            xSumSquared += x[i] * x[i];
            ySumSquared += y[i] * y[i];
        }

        double result = dot / (Math.Sqrt(xSumSquared) * Math.Sqrt(ySumSquared));
        return (float)result;
    }
}

public readonly struct VectorbaseEntry
{
    readonly ReadOnlyMemory<float> _vector;
    readonly int? _id;
    readonly BinaryData _data;

    public VectorbaseEntry(ReadOnlyMemory<float> vector, BinaryData data, int? id = default)
    {
        _vector = vector;
        _data = data;
        _id = id;
    }

    public BinaryData Data => _data;
    public ReadOnlyMemory<float> Vector => _vector;
    public int? Id => _id;
}



// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Generic;
using System;

namespace OpenAI.Embeddings;

internal class MemoryVectorbaseStore : VectorbaseStore
{
    readonly List<VectorbaseEntry> _entries = new List<VectorbaseEntry>();

    public override IEnumerable<VectorbaseEntry> Find(ReadOnlyMemory<float> vector, FindOptions options)
    {
        lock (_entries)
        {
            var distances = new List<(float Distance, int Index)>(_entries.Count);
            for (int index = 0; index<_entries.Count; index++)
            {
                VectorbaseEntry entry = _entries[index];
                ReadOnlyMemory<float> dbVector = entry.Vector;
                float distance = 1.0f - CosineSimilarity(dbVector.Span, vector.Span);
                distances.Add((distance, index));
            }
            distances.Sort(((float d1, int i1) v1, (float d2, int i2) v2) => v1.d1.CompareTo(v2.d2));

            var results = new List<VectorbaseEntry>(options.MaxEntries);
            var top = Math.Min(options.MaxEntries, distances.Count);
            for (int i = 0; i<top; i++)
            {
                var distance = distances[i].Distance;
                if (distance > options.Threshold) break;
                var index = distances[i].Index;
                results.Add(_entries[index]);
            }
            return results;
        }
    }
    public override int Add(VectorbaseEntry entry)
    {
        lock (_entries)
        {
            var id = _entries.Count;
            var newEntry = new VectorbaseEntry(entry.Vector, entry.Data, id);
            _entries.Add(newEntry);
            return id;
        }
    }

    public override void Add(IReadOnlyList<VectorbaseEntry> entries)
    {
        foreach (var entry in entries)
        {
            Add(entry);
        }
    }
}


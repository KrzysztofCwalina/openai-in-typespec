﻿using NUnit.Framework;
using OpenAI.Embeddings;
using System;
using System.Collections.Generic;

namespace OpenAI.Samples
{
    public partial class EmbeddingSamples
    {
        [Test]
        [Ignore("Compilation validation only")]
        public void Sample03_MultipleEmbeddings()
        {
            EmbeddingClient client = new("text-embedding-3-small", Environment.GetEnvironmentVariable("OpenAIClient_KEY"));

            string category = "Luxury";
            string description =
                "Best hotel in town if you like luxury hotels. They have an amazing infinity pool, a spa,"
                + " and a really helpful concierge. The location is perfect -- right downtown, close to all "
                + " the tourist attractions. We highly recommend this hotel.";
            List<string> inputs = [category, description];

            EmbeddingCollection collection = client.GenerateEmbeddings(inputs);

            foreach (Embedding embedding in collection)
            {
                ReadOnlyMemory<float> vector = embedding.Vector;

                Console.WriteLine($"Dimension: {vector.Length}");
                Console.WriteLine($"Floats: ");
                for (int i = 0; i < vector.Length; i++)
                {
                    Console.WriteLine($"  [{i}] = {vector.Span[i]}");
                }

                Console.WriteLine();
            }
        }
    }
}

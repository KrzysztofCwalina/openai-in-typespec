﻿using NUnit.Framework;
using OpenAI.Assistants;
using System;

namespace OpenAI.Samples
{
    public partial class AssistantSamples
    {
        [Test]
        [Ignore("Compilation validation only")]
        public void Sample02_ListAssistantsWithPagination()
        {
            AssistantClient client = new(Environment.GetEnvironmentVariable("OpenAIClient_KEY"));

            string latestId = null;
            bool continueQuery = true;
            int count = 0;

            while (continueQuery)
            {
                ListQueryPage<Assistant> pagedAssistants = client.GetAssistants(previousAssistantId: latestId);

                foreach (Assistant assistant in pagedAssistants)
                {
                    Console.WriteLine($"[{count,3}] {assistant.Id} {assistant.CreatedAt:s} {assistant.Name}");

                    latestId = assistant.Id;
                    count++;
                }

                continueQuery = pagedAssistants.HasMore;
            }
        }
    }
}

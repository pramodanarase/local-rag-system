package com.rag;

import com.rag.config.Config;
import com.rag.model.Document;
import com.rag.model.TextChunk;
import com.rag.service.DocumentProcessor;
import com.rag.service.EmbeddingService;
import com.rag.service.LMStudioClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class QueryRepository {
    private static final Logger logger = LoggerFactory.getLogger(QueryRepository.class);

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Usage: java -cp target/local-rag-system-1.0-SNAPSHOT-jar-with-dependencies.jar com.rag.QueryRepository \"Your question about the codebase?\"");
            System.exit(1);
        }

        String question = String.join(" ", args);
        logger.info("Processing question: {}", question);

        try (
            EmbeddingService embeddingService = new EmbeddingService();
            LMStudioClient lmStudioClient = new LMStudioClient()
        ) {
            // Get query embedding
            float[] queryEmbedding = embeddingService.generateEmbedding(question);

            // Process repository to get documents
            DocumentProcessor documentProcessor = new DocumentProcessor(embeddingService);
            List<Document> documents = documentProcessor.processRepository(Config.REPO_PATH);

            // Find most relevant chunks
            List<RelevantChunk> relevantChunks = new ArrayList<>();
            for (Document doc : documents) {
                for (TextChunk chunk : doc.getChunks()) {
                    float similarity = cosineSimilarity(queryEmbedding, chunk.getEmbedding());
                    relevantChunks.add(new RelevantChunk(doc, chunk, similarity));
                }
            }

            // Sort by similarity and take top K
            List<String> contextChunks = relevantChunks.stream()
                .sorted(Comparator.comparing(RelevantChunk::similarity).reversed())
                .limit(Config.TOP_K)
                .map(rc -> formatChunk(rc.document(), rc.chunk()))
                .collect(Collectors.toList());

            // Get answer from LM Studio
            String answer = lmStudioClient.complete(question, contextChunks);

            // Print results
            System.out.println("\nQuestion: " + question);
            System.out.println("\nAnswer: " + answer);

        } catch (Exception e) {
            logger.error("Error processing query", e);
            System.exit(1);
        }
    }

    private static String formatChunk(Document doc, TextChunk chunk) {
        return String.format("File: %s\n%s", doc.getPath(), chunk.getText());
    }

    private static float cosineSimilarity(float[] v1, float[] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("Vectors must have same length");
        }

        float dotProduct = 0;
        float norm1 = 0;
        float norm2 = 0;

        for (int i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        return (float) (dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2)));
    }

    private record RelevantChunk(Document document, TextChunk chunk, float similarity) {}
} 
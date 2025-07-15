package com.rag.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class EmbeddingService implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(EmbeddingService.class);
    private static final int EMBEDDING_SIZE = 384; // Standard size for small embedding models

    public EmbeddingService() {
        logger.info("Initialized simplified embedding service");
    }

    public float[] generateEmbedding(String text) {
        // Simple embedding generation based on character frequencies
        float[] embedding = new float[EMBEDDING_SIZE];
        
        // Initialize with small random values
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            embedding[i] = (float) (Math.random() * 0.1);
        }
        
        // Add character frequency information
        for (char c : text.toCharArray()) {
            int index = Math.abs(c % EMBEDDING_SIZE);
            embedding[index] += 0.1f;
        }
        
        // Normalize the embedding
        float sum = 0;
        for (float value : embedding) {
            sum += value * value;
        }
        float norm = (float) Math.sqrt(sum);
        
        for (int i = 0; i < EMBEDDING_SIZE; i++) {
            embedding[i] /= norm;
        }
        
        logger.debug("Generated embedding of size: {}", EMBEDDING_SIZE);
        return embedding;
    }

    @Override
    public void close() {
        // No resources to close in this simplified version
    }
} 
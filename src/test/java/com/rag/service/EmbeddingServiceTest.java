package com.rag.service;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import ai.djl.translate.TranslateException;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingServiceTest {
    private EmbeddingService service;
    
    @BeforeEach
    void setUp() throws IOException {
        service = new EmbeddingService();
    }
    
    @Test
    void testEmbeddingDimension() throws IOException, TranslateException {
        float[] embedding = service.generateEmbeddings("Test text");
        assertEquals(384, embedding.length, "Embedding should have correct dimension");
    }
    
    @Test
    void testNonEmptyEmbedding() throws IOException, TranslateException {
        float[] embedding = service.generateEmbeddings("Another test");
        boolean hasNonZero = false;
        for (float value : embedding) {
            if (value != 0) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "Embedding should contain non-zero values");
    }
    
    @Test
    void testConsistentDimension() throws IOException, TranslateException {
        float[] embedding1 = service.generateEmbeddings("First text");
        float[] embedding2 = service.generateEmbeddings("Second text");
        assertEquals(embedding1.length, embedding2.length, 
            "Embeddings of different texts should have same dimension");
    }
    
    @Test
    void testEmptyString() throws IOException, TranslateException {
        float[] embedding = service.generateEmbeddings("");
        assertNotNull(embedding, "Should handle empty string");
        assertEquals(384, embedding.length, "Empty string embedding should have correct dimension");
    }
    
    @Test
    void testLongText() throws IOException, TranslateException {
        String longText = "a".repeat(10000);
        float[] embedding = service.generateEmbeddings(longText);
        assertEquals(384, embedding.length, 
            "Long text embedding should have correct dimension");
    }
    
    @Test
    void testSpecialCharacters() throws IOException, TranslateException {
        String specialText = "!@#$%^&*()_+-=[]{}|;:'\",.<>?/\\";
        float[] embedding = service.generateEmbeddings(specialText);
        assertEquals(384, embedding.length, 
            "Special characters embedding should have correct dimension");
    }
    
    @AfterEach
    void tearDown() throws IOException {
        if (service != null) {
            service.close();
        }
    }
} 
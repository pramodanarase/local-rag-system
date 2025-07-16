package com.rag.service;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.rag.vectordb.LuceneVectorStore;
import com.rag.vectordb.SearchResult;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class LuceneVectorStoreTest {
    private static final Logger logger = LoggerFactory.getLogger(LuceneVectorStoreTest.class);
    private Directory directory;
    private LuceneVectorStore vectorStore;

    @BeforeEach
    void setUp() throws IOException {
        // Use in-memory directory for testing
        directory = new ByteBuffersDirectory();
        vectorStore = new LuceneVectorStore(directory);
    }

    @AfterEach
    void tearDown() throws IOException {
        if (vectorStore != null) {
            vectorStore.close();
        }
        if (directory != null) {
            directory.close();
        }
    }

    @Test
    void testAddAndRetrieveVector() throws IOException {
        // Test vector and metadata
        float[] vector = {0.1f, 0.2f, 0.3f, 0.4f};
        String id = "test1";
        String content = "This is a test document";
        String source = "test.txt";

        // Add vector to store
        vectorStore.addVector(vector, id, content, source);
        vectorStore.flush(); // Ensure vector is committed and searchable

        // Search for similar vectors
        List<SearchResult> results = vectorStore.findSimilar(vector, 1);

        // Verify results
        assertFalse(results.isEmpty(), "Should find at least one result");
        SearchResult result = results.get(0);
        assertEquals(id, result.id());
        assertEquals(content, result.content());
        assertEquals(source, result.source());
        assertTrue(result.score() > 0.99, "Score should be very high for exact match");
    }

    @Test
    void testMultipleVectors() throws IOException {
        // Add multiple test vectors with more distinct values
        float[] vector1 = {1.0f, 0.0f, 0.0f, 0.0f};  // Unit vector in x direction
        float[] vector2 = {0.0f, 1.0f, 0.0f, 0.0f};  // Unit vector in y direction
        float[] vector3 = {0.0f, 0.0f, 1.0f, 0.0f};  // Unit vector in z direction

        logger.debug("Vector1: {}", Arrays.toString(vector1));
        logger.debug("Vector2: {}", Arrays.toString(vector2));
        logger.debug("Vector3: {}", Arrays.toString(vector3));

        vectorStore.addVector(vector1, "doc1", "First document", "source1.txt");
        vectorStore.addVector(vector2, "doc2", "Second document", "source2.txt");
        vectorStore.addVector(vector3, "doc3", "Third document", "source3.txt");
        vectorStore.flush(); // Ensure all vectors are committed and searchable

        // Search with vector1
        List<SearchResult> results = vectorStore.findSimilar(vector1, 3);

        assertEquals(3, results.size(), "Should return all three results");
        
        // Log all results
        for (SearchResult result : results) {
            logger.debug("Result: {} (score: {})", result.id(), result.score());
        }
        
        // First result should be vector1 itself
        assertEquals("doc1", results.get(0).id(), 
            String.format("First result should be doc1, but was %s with score %f", 
                results.get(0).id(), results.get(0).score()));
        
        // Since vectors are orthogonal, they should have zero similarity
        assertTrue(Math.abs(results.get(1).score()) < 0.0001, 
            String.format("Second result (%s) should have near-zero score but was %f", 
                results.get(1).id(), results.get(1).score()));
        assertTrue(Math.abs(results.get(2).score()) < 0.0001, 
            String.format("Third result (%s) should have near-zero score but was %f", 
                results.get(2).id(), results.get(2).score()));
    }

    @Test
    void testEmptyStore() throws IOException {
        float[] queryVector = {0.1f, 0.2f, 0.3f, 0.4f};
        List<SearchResult> results = vectorStore.findSimilar(queryVector, 1);
        
        assertTrue(results.isEmpty(), "Empty store should return no results");
    }

    @Test
    void testNormalization() throws IOException {
        // Test vectors with different magnitudes but same direction
        float[] vector1 = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] vector2 = {0.2f, 0.4f, 0.6f, 0.8f}; // Same direction as vector1, double magnitude

        vectorStore.addVector(vector1, "doc1", "First document", "source1.txt");
        vectorStore.addVector(vector2, "doc2", "Second document", "source2.txt");
        vectorStore.flush(); // Ensure vectors are committed and searchable

        List<SearchResult> results = vectorStore.findSimilar(vector1, 2);

        assertEquals(2, results.size(), "Should return both results");
        assertTrue(Math.abs(results.get(0).score() - results.get(1).score()) < 0.0001,
            String.format("Normalized vectors in same direction should have very similar scores: %f vs %f",
                results.get(0).score(), results.get(1).score()));
    }
} 
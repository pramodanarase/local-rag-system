package com.rag.vectordb;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Vector store implementation using Apache Lucene.
 * Stores document vectors and metadata, supports similarity search.
 */
public class LuceneVectorStore implements Closeable {
    private static final Logger logger = LoggerFactory.getLogger(LuceneVectorStore.class);
    private static final String VECTOR_FIELD = "vector";
    private static final String ID_FIELD = "id";
    private static final String CONTENT_FIELD = "content";
    private static final String SOURCE_FIELD = "source";
    private static final int COMMIT_INTERVAL = 100;

    private final Directory directory;
    private final IndexWriter writer;
    private final SearcherManager searcherManager;
    private int documentsAdded;

    /**
     * Creates a new vector store using the provided Lucene Directory.
     *
     * @param directory Lucene Directory to use for storage
     * @throws IOException if index initialization fails
     */
    public LuceneVectorStore(Directory directory) throws IOException {
        this.directory = directory;
        
        // Check if index exists
        boolean indexExists = DirectoryReader.indexExists(directory);
        
        // Create empty index if it doesn't exist
        if (!indexExists) {
            IndexWriterConfig createConfig = new IndexWriterConfig(new StandardAnalyzer());
            createConfig.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
            try (IndexWriter createWriter = new IndexWriter(directory, createConfig)) {
                // Just create the index, no need to add anything
                createWriter.commit();
            }
            logger.debug("Created new empty index in directory: {}", directory);
        }
        
        // Now open in append mode
        IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
        config.setOpenMode(IndexWriterConfig.OpenMode.APPEND);
        config.setRAMBufferSizeMB(256.0); // Increased for better performance
        config.setMaxBufferedDocs(1000); // Increased for better batching
        config.setUseCompoundFile(true); // Use compound file format for better persistence
        config.setCommitOnClose(true); // Ensure changes are committed on close
        
        this.writer = new IndexWriter(directory, config);
        this.searcherManager = new SearcherManager(writer, null);
        this.documentsAdded = 0;
        
        logger.debug("Opened index in APPEND mode: {}", directory);
    }

    /**
     * Adds a vector with associated metadata to the store.
     *
     * @param vector The vector to store
     * @param id Unique identifier for the document
     * @param content Text content associated with the vector
     * @param source Source identifier (e.g., filename)
     * @throws IOException if writing to the index fails
     */
    public void addVector(float[] vector, String id, String content, String source) throws IOException {
        // Create a copy of the vector to avoid modifying the input
        float[] vectorCopy = vector.clone();
        
        // Normalize the vector before storing
        float magnitude = magnitude(vectorCopy);
        logger.debug("Adding vector {} with magnitude {}", id, magnitude);
        
        if (magnitude > 0) {
            vectorCopy = normalize(vectorCopy, magnitude);
            logger.debug("Normalized vector {}: {}", id, vectorToString(vectorCopy));
        }
        
        Document doc = new Document();
        
        // Store the vector
        doc.add(new StoredField(VECTOR_FIELD, VectorUtil.encode(vectorCopy)));
        
        // Store metadata
        doc.add(new StringField(ID_FIELD, id, Field.Store.YES));
        doc.add(new TextField(CONTENT_FIELD, content, Field.Store.YES));
        doc.add(new StringField(SOURCE_FIELD, source, Field.Store.YES));
        
        writer.addDocument(doc);
        documentsAdded++;

        // Commit and refresh based on interval
        if (documentsAdded % COMMIT_INTERVAL == 0) {
            flush();
        }
    }

    /**
     * Forces a commit of any buffered documents and refreshes the searcher.
     * Call this when you need to ensure all documents are searchable.
     *
     * @throws IOException if the commit or refresh fails
     */
    public void flush() throws IOException {
        writer.commit();
        searcherManager.maybeRefresh();
        logger.debug("Committed {} documents to index", documentsAdded);
    }

    /**
     * Finds similar vectors using cosine similarity.
     *
     * @param queryVector The query vector
     * @param k Number of results to return
     * @return List of search results, ordered by decreasing similarity
     * @throws IOException if reading from the index fails
     */
    public List<SearchResult> findSimilar(float[] queryVector, int k) throws IOException {
        // Create a copy of the query vector to avoid modifying the input
        float[] queryVectorCopy = queryVector.clone();
        
        IndexSearcher searcher = searcherManager.acquire();
        try {
            // Normalize query vector
            float queryMagnitude = magnitude(queryVectorCopy);
            logger.debug("Query vector magnitude: {}", queryMagnitude);
            
            if (queryMagnitude == 0) {
                throw new IllegalArgumentException("Query vector magnitude is 0");
            }
            queryVectorCopy = normalize(queryVectorCopy, queryMagnitude);
            logger.debug("Normalized query vector: {}", vectorToString(queryVectorCopy));

            // Search all documents
            TopDocs topDocs = searcher.search(new MatchAllDocsQuery(), Integer.MAX_VALUE);
            
            // Calculate similarities and keep top k
            PriorityQueue<SearchResult> results = new PriorityQueue<>(k, 
                (a, b) -> Float.compare(a.score(), b.score())); // Changed to keep highest scores

            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                Document doc = searcher.storedFields().document(scoreDoc.doc);
                String docId = doc.get(ID_FIELD);
                
                byte[] encodedVector = doc.getBinaryValue(VECTOR_FIELD).bytes;
                float[] docVector = VectorUtil.decode(encodedVector);
                logger.debug("Document vector {}: {}", docId, vectorToString(docVector));
                
                // Calculate cosine similarity (dot product of normalized vectors)
                float similarity = dotProduct(queryVectorCopy, docVector);
                logger.debug("Similarity between query and {}: {}", docId, similarity);
                
                // Ensure similarity is in [-1, 1] range
                similarity = Math.max(-1.0f, Math.min(1.0f, similarity));
                
                // Add a small bias to exact matches
                if (similarity > 0.9999f) {
                    similarity = 1.0f;
                }
                
                if (results.size() < k || similarity > results.peek().score()) {
                    if (results.size() == k) {
                        results.poll(); // Remove lowest score
                    }
                    results.offer(new SearchResult(
                        docId,
                        doc.get(CONTENT_FIELD),
                        doc.get(SOURCE_FIELD),
                        similarity
                    ));
                }
            }

            // Convert to sorted list
            ArrayList<SearchResult> sortedResults = new ArrayList<>(results.size());
            while (!results.isEmpty()) {
                sortedResults.add(0, results.poll());
            }
            
            // Log final results
            logger.debug("Final results:");
            for (SearchResult result : sortedResults) {
                logger.debug("  {} (score: {})", result.id(), result.score());
            }
            
            return sortedResults;

        } finally {
            searcherManager.release(searcher);
        }
    }

    @Override
    public void close() throws IOException {
        try {
            // Commit any remaining documents
            if (documentsAdded > 0) {
                flush();
            }
            
            // Close searcher manager first
            if (searcherManager != null) {
                searcherManager.close();
            }
            
            // Close writer with commit
            if (writer != null) {
                writer.close();
            }
            
            // Don't close the directory - let the caller handle that
            logger.info("Closed index writer and searcher manager. Documents added: {}", documentsAdded);
        } catch (Exception e) {
            logger.error("Error closing index", e);
            throw e;
        }
    }

    // Utility methods for vector operations
    
    private float magnitude(float[] vector) {
        float sum = 0;
        for (float v : vector) {
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }
    
    private float[] normalize(float[] vector, float magnitude) {
        float[] normalized = vector.clone();
        for (int i = 0; i < normalized.length; i++) {
            normalized[i] /= magnitude;
        }
        return normalized;
    }
    
    private float dotProduct(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
    
    private String vectorToString(float[] vector) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < Math.min(3, vector.length); i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.3f", vector[i]));
        }
        if (vector.length > 3) sb.append(", ...");
        sb.append("]");
        return sb.toString();
    }
    
    private static class VectorUtil {
        static byte[] encode(float[] vector) {
            byte[] bytes = new byte[vector.length * 4];
            for (int i = 0; i < vector.length; i++) {
                int bits = Float.floatToIntBits(vector[i]);
                bytes[i*4] = (byte) (bits >> 24);
                bytes[i*4 + 1] = (byte) (bits >> 16);
                bytes[i*4 + 2] = (byte) (bits >> 8);
                bytes[i*4 + 3] = (byte) bits;
            }
            return bytes;
        }

        static float[] decode(byte[] bytes) {
            float[] vector = new float[bytes.length / 4];
            for (int i = 0; i < vector.length; i++) {
                int bits = ((bytes[i*4] & 0xFF) << 24) |
                          ((bytes[i*4 + 1] & 0xFF) << 16) |
                          ((bytes[i*4 + 2] & 0xFF) << 8) |
                          (bytes[i*4 + 3] & 0xFF);
                vector[i] = Float.intBitsToFloat(bits);
            }
            return vector;
        }
    }
} 
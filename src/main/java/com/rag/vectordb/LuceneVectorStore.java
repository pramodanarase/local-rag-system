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
        
        // Store the vector using KnnFloatVectorField
        doc.add(new KnnFloatVectorField(VECTOR_FIELD, vectorCopy, VectorSimilarityFunction.COSINE));
    
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
     * Finds similar vectors using KNN vector search with cosine similarity.
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

            // Create KNN vector query
            Query knnQuery = new KnnFloatVectorQuery(VECTOR_FIELD, queryVectorCopy, k);
            TopDocs topDocs = searcher.search(knnQuery, k);
            
            List<SearchResult> results = new ArrayList<>();
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                Document doc = searcher.storedFields().document(scoreDoc.doc);
                results.add(new SearchResult(
                    doc.get(ID_FIELD),
                    doc.get(CONTENT_FIELD),
                    doc.get(SOURCE_FIELD),
                    scoreDoc.score
                ));
            }
            
            // Log final results
            logger.debug("Final results:");
            for (SearchResult result : results) {
                logger.debug("  {} (score: {})", result.id(), result.score());
            }
            
            return results;

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
    
    private String vectorToString(float[] vector) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < vector.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.4f", vector[i]));
        }
        sb.append("]");
        return sb.toString();
    }
} 
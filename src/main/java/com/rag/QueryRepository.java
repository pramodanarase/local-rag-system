package com.rag;

import com.rag.config.Config;
import com.rag.model.TextChunk;
import com.rag.service.EmbeddingService;
import com.rag.vectordb.LuceneVectorStore;
import com.rag.vectordb.SearchResult;

import ai.djl.translate.TranslateException;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class QueryRepository implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(QueryRepository.class);
    
    private final LuceneVectorStore vectorStore;
    private final EmbeddingService embeddingService;
    
    public QueryRepository(EmbeddingService embeddingService) throws IOException {
        this(embeddingService, new LuceneVectorStore(new NIOFSDirectory(Config.INDEX_PATH)));
    }

    public QueryRepository(EmbeddingService embeddingService, LuceneVectorStore vectorStore) {
        this.embeddingService = embeddingService;
        this.vectorStore = vectorStore;
        
        logger.info("Initialized QueryRepository with index at: {}", Config.INDEX_PATH);
    }
    
    public List<TextChunk> query(String text, int k) throws IOException {
        // Get embedding for query text
        float[] queryEmbedding;
        try {
            queryEmbedding = embeddingService.generateEmbeddings(text);
        } catch (TranslateException e) {
            throw new IOException("Failed to generate embeddings", e);
        }
        
        // Search for similar chunks
        List<SearchResult> results = vectorStore.findSimilar(queryEmbedding, k);
        
        // Convert search results to text chunks
        List<TextChunk> chunks = new ArrayList<>();
        for (SearchResult result : results) {
            chunks.add(new TextChunk(
                result.content(),
                null, // We don't need embedding in results
                0,   // Position info not needed for results
                result.content().length()
            ));
        }
        
        return chunks;
    }
    
    @Override
    public void close() throws IOException {
        vectorStore.close();
    }
} 
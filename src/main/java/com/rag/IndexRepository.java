package com.rag;

import com.rag.config.Config;
import com.rag.model.Document;
import com.rag.service.DocumentProcessor;
import com.rag.service.EmbeddingService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class IndexRepository {
    private static final Logger logger = LoggerFactory.getLogger(IndexRepository.class);

    public static void main(String[] args) {
        // Create data directory if it doesn't exist
        try {
            Files.createDirectories(Config.INDEX_PATH);
        } catch (Exception e) {
            logger.error("Failed to create index directory", e);
            System.exit(1);
        }

        // Initialize services
        try (EmbeddingService embeddingService = new EmbeddingService()) {
            DocumentProcessor documentProcessor = new DocumentProcessor(embeddingService);
            
            // Process repository
            try {
                logger.info("Starting repository indexing from: {}", Config.REPO_PATH);
                List<Document> documents = documentProcessor.processRepository(Config.REPO_PATH);
                logger.info("Processed {} documents", documents.size());
                
                // Save documents to index
                // Note: In a real implementation, you'd want to save these to a vector store
                // For this example, we're just counting the documents
                for (Document doc : documents) {
                    logger.info("Indexed: {} with {} chunks", doc.getPath(), doc.getChunks().size());
                }
                
                logger.info("Indexing completed successfully");
            } catch (Exception e) {
                logger.error("Failed to process repository", e);
                System.exit(1);
            }
        } catch (Exception e) {
            logger.error("Failed to initialize services", e);
            System.exit(1);
        }
    }
} 
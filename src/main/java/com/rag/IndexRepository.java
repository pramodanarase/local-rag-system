package com.rag;

import com.rag.config.Config;
import com.rag.model.Document;
import com.rag.model.TextChunk;
import com.rag.service.DocumentProcessor;
import com.rag.service.EmbeddingService;
import com.rag.vectordb.LuceneVectorStore;

import ai.djl.translate.TranslateException;

import org.apache.lucene.store.NIOFSDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

public class IndexRepository implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(IndexRepository.class);
    
    private final LuceneVectorStore vectorStore;
    private final DocumentProcessor documentProcessor;
    private final EmbeddingService embeddingService;
    
    public IndexRepository(DocumentProcessor documentProcessor, EmbeddingService embeddingService) throws IOException {
        this(documentProcessor, embeddingService, new LuceneVectorStore(new NIOFSDirectory(Config.INDEX_PATH)));
    }

    public IndexRepository(DocumentProcessor documentProcessor, EmbeddingService embeddingService, LuceneVectorStore vectorStore) {
        this.documentProcessor = documentProcessor;
        this.embeddingService = embeddingService;
        this.vectorStore = vectorStore;
        
        logger.info("Initialized IndexRepository with index at: {}", Config.INDEX_PATH);
    }
    
    public void indexDirectory(Path directory) throws IOException {
        try (Stream<Path> files = Files.walk(directory)) {
            files.filter(Files::isRegularFile)
                .filter(this::isTextFile)
                .filter(this::isNotExcluded)
                .forEach(file -> {
                    try {
                        indexFile(file);
                    } catch (IOException e) {
                        logger.error("Error indexing file: {}", file, e);
                    } catch (OutOfMemoryError e) {
                        logger.error("Out of memory while indexing file: {}. Skipping.", file);
                    } catch (TranslateException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                });
        }
    }

    private boolean isTextFile(Path file) {
        String fileName = file.getFileName().toString().toLowerCase();
        return Config.INCLUDE_EXTENSIONS.stream()
            .anyMatch(ext -> fileName.endsWith(ext.toLowerCase()));
    }

    private boolean isNotExcluded(Path file) {
        String path = file.toString().toLowerCase();
        return Config.EXCLUDE_DIRS.stream()
            .noneMatch(dir -> path.contains("/" + dir.toLowerCase() + "/"));
    }
    
    public void indexFile(Path file) throws IOException, TranslateException {
        logger.info("Indexing file: {}", file);
        
        // Process the document into chunks
        Document doc = documentProcessor.processFile(file);
        
        // Index each chunk
        for (TextChunk chunk : doc.getChunks()) {
            // Get embedding for the chunk
            float[] embedding = embeddingService.generateEmbeddings(chunk.getText());
            
            // Add to vector store
            vectorStore.addVector(
                embedding,
                file.toString() + "#" + chunk.getStartPosition(),
                chunk.getText(),
                file.toString()
            );
        }
        
        // Ensure all chunks are committed and searchable
        vectorStore.flush();
        
        logger.debug("Indexed {} chunks from file: {}", doc.getChunks().size(), file);
    }
    
    @Override
    public void close() throws IOException {
        vectorStore.close();
    }
} 
package com.rag.service;

import com.rag.config.Config;
import com.rag.model.Document;
import com.rag.model.TextChunk;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class DocumentProcessor {
    private static final Logger logger = LoggerFactory.getLogger(DocumentProcessor.class);
    private final EmbeddingService embeddingService;

    public DocumentProcessor(EmbeddingService embeddingService) {
        this.embeddingService = embeddingService;
    }

    public List<Document> processRepository(String repoPath) throws IOException {
        Path path = Paths.get(repoPath);
        List<Document> documents = new ArrayList<>();

        Files.walkFileTree(path, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                if (shouldProcessFile(file)) {
                    try {
                        Document doc = processFile(file);
                        documents.add(doc);
                        logger.debug("Processed file: {}", file);
                    } catch (Exception e) {
                        logger.error("Error processing file: {}", file, e);
                    }
                }
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) {
                return shouldExcludeDirectory(dir) ? 
                    FileVisitResult.SKIP_SUBTREE : FileVisitResult.CONTINUE;
            }
        });

        return documents;
    }

    private boolean shouldProcessFile(Path file) {
        String extension = getFileExtension(file);
        return Config.INCLUDE_EXTENSIONS.contains(extension);
    }

    private boolean shouldExcludeDirectory(Path dir) {
        return Config.EXCLUDE_DIRS.contains(dir.getFileName().toString());
    }

    private String getFileExtension(Path file) {
        String name = file.getFileName().toString();
        int lastDotIndex = name.lastIndexOf('.');
        return lastDotIndex > 0 ? name.substring(lastDotIndex) : "";
    }

    private Document processFile(Path file) throws IOException {
        String content = Files.readString(file);
        Document document = new Document(file, content);
        
        // Create chunks
        List<String> chunks = createChunks(content);
        
        // Process each chunk
        for (int i = 0; i < chunks.size(); i++) {
            String chunkText = chunks.get(i);
            float[] embedding = embeddingService.generateEmbedding(chunkText);
            
            int startPos = i * (Config.CHUNK_SIZE - Config.CHUNK_OVERLAP);
            int endPos = startPos + chunkText.length();
            
            TextChunk chunk = new TextChunk(chunkText, embedding, startPos, endPos);
            document.addChunk(chunk);
        }
        
        return document;
    }

    private List<String> createChunks(String content) {
        List<String> chunks = new ArrayList<>();
        int length = content.length();
        int chunkSize = Config.CHUNK_SIZE;
        int overlap = Config.CHUNK_OVERLAP;
        
        for (int i = 0; i < length; i += chunkSize - overlap) {
            int end = Math.min(i + chunkSize, length);
            chunks.add(content.substring(i, end));
        }
        
        return chunks;
    }
} 
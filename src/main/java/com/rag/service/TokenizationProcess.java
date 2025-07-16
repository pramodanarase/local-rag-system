package com.rag.service;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import com.rag.config.Config;
import com.rag.model.Document;
import com.rag.model.TextChunk;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class TokenizationProcess implements Closeable {
    private static final Logger logger = LoggerFactory.getLogger(TokenizationProcess.class);
    private static final int MAX_FILE_SIZE = 100 * 1024; // 100KB
    private static final int MAX_BUFFER_SIZE = 4 * 1024; // 4KB
    private static final int MAX_TOKENS = 512; // Maximum tokens per chunk for all-MiniLM-L6-v2

    private final HuggingFaceTokenizer tokenizer;

    public TokenizationProcess() throws IOException {
        // Initialize the tokenizer with options
        Map<String, String> options = Map.of(
                "truncation", "true",
                "maxLength", String.valueOf(MAX_TOKENS));

        this.tokenizer = HuggingFaceTokenizer.newInstance(
                Paths.get(Config.LOCAL_MODEL_PATH),
                options
        );
        logger.info("Initialized HuggingFace tokenizer from {}", Config.LOCAL_MODEL_PATH);
    }

    /**
     * Process a file into a Document with overlapping text chunks.
     *
     * @param file The file to process
     * @return A Document containing the processed text chunks
     * @throws IOException if reading the file fails
     */
    public Document processFile(Path file) throws IOException {
        // Check file size first
        long fileSize = Files.size(file);
        if (fileSize > MAX_FILE_SIZE) {
            logger.warn("Skipping large file ({}KB): {}", fileSize / 1024, file);
            return new Document(file, List.of());
        }

        List<TextChunk> chunks = new ArrayList<>();
        StringBuilder buffer = new StringBuilder(MAX_BUFFER_SIZE);
        int position = 0;

        // Process file line by line to avoid loading entire file into memory
        try (BufferedReader reader = Files.newBufferedReader(file)) {
            String line;
            while ((line = reader.readLine()) != null) {
                // Skip empty lines and very long lines
                if (line.trim().isEmpty() || line.length() > MAX_BUFFER_SIZE) {
                    continue;
                }

                buffer.append(line).append('\n');
                
                // Process buffer when it exceeds chunk size
                if (buffer.length() >= Config.CHUNK_SIZE) {
                    chunks.addAll(createChunks(buffer.toString(), position));
                    position += buffer.length() - Config.CHUNK_OVERLAP;
                    
                    // Keep overlap portion for next chunk
                    if (buffer.length() > Config.CHUNK_OVERLAP) {
                        buffer.delete(0, buffer.length() - Config.CHUNK_OVERLAP);
                    } else {
                        buffer.setLength(0);
                    }
                }

                // Clear buffer if it gets too large
                if (buffer.length() > MAX_BUFFER_SIZE) {
                    buffer.setLength(0);
                }
            }
            
            // Process remaining text
            if (buffer.length() > 0) {
                chunks.addAll(createChunks(buffer.toString(), position));
            }
        }

        return new Document(file, chunks);
    }
    
    /**
     * Creates overlapping chunks from the input text using HuggingFace tokenization.
     *
     * @param text The text to chunk
     * @param startPosition The starting position in the original file
     * @return List of text chunks
     */
    private List<TextChunk> createChunks(String text, int startPosition) {
        List<TextChunk> chunks = new ArrayList<>();
        
        // First, split text into sentences for better chunking
        String[] sentences = text.split("(?<=[.!?])\\s+");
        StringBuilder currentChunk = new StringBuilder();
        int currentPosition = startPosition;
        
        for (String sentence : sentences) {
            // Get token count for this sentence
            long sentenceTokens = tokenizer.encode(sentence).getIds().length;
            
            // If adding this sentence would exceed max tokens, create a new chunk
            long currentTokens = tokenizer.encode(currentChunk.toString()).getIds().length;
            if (currentTokens + sentenceTokens > MAX_TOKENS && currentChunk.length() > 0) {
                // Create chunk from current buffer
                String chunkText = currentChunk.toString().trim();
                if (!chunkText.isEmpty()) {
                    chunks.add(new TextChunk(
                            chunkText,
                            null,
                            currentPosition,
                            currentPosition + chunkText.length()
                    ));
                }
                
                // Start new chunk with overlap
                int overlapStart = Math.max(0, currentChunk.length() - Config.CHUNK_OVERLAP);
                String overlap = currentChunk.substring(overlapStart);
                currentChunk.setLength(0);
                currentChunk.append(overlap);
                currentPosition += chunkText.length() - overlap.length();
            }
            
            // Add the sentence to current chunk
            currentChunk.append(sentence).append(" ");
        }
        
        // Add final chunk if there's anything left
        String finalChunk = currentChunk.toString().trim();
        if (!finalChunk.isEmpty()) {
            chunks.add(new TextChunk(
                    finalChunk,
                    null,
                    currentPosition,
                    currentPosition + finalChunk.length()
            ));
        }
        
        return chunks;
    }

    @Override
    public void close() throws IOException {
        if (tokenizer != null) {
            tokenizer.close();
        }
    }
} 
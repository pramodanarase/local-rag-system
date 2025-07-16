package com.rag.model;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Document {
    private final Path path;
    private final List<TextChunk> chunks;
    
    public Document(Path path, List<TextChunk> chunks) {
        this.path = path;
        this.chunks = new ArrayList<>(chunks);
    }
    
    public Path getPath() {
        return path;
    }
    
    public List<TextChunk> getChunks() {
        return Collections.unmodifiableList(chunks);
    }
} 
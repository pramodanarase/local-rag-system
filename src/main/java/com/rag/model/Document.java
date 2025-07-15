package com.rag.model;

import java.nio.file.Path;
import java.util.List;
import java.util.ArrayList;

public class Document {
    private final Path path;
    private final String content;
    private final List<TextChunk> chunks;

    public Document(Path path, String content) {
        this.path = path;
        this.content = content;
        this.chunks = new ArrayList<>();
    }

    public Path getPath() {
        return path;
    }

    public String getContent() {
        return content;
    }

    public List<TextChunk> getChunks() {
        return chunks;
    }

    public void addChunk(TextChunk chunk) {
        chunks.add(chunk);
    }

    @Override
    public String toString() {
        return String.format("Document(path=%s, chunks=%d)", path, chunks.size());
    }
} 
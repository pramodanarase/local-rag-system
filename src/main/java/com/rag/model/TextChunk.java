package com.rag.model;

import java.util.Arrays;

public class TextChunk {
    private final String text;
    private final float[] embedding;
    private final int startPosition;
    private final int endPosition;

    public TextChunk(String text, float[] embedding, int startPosition, int endPosition) {
        this.text = text;
        this.embedding = embedding;
        this.startPosition = startPosition;
        this.endPosition = endPosition;
    }

    public String getText() {
        return text;
    }

    public float[] getEmbedding() {
        return embedding;
    }

    public int getStartPosition() {
        return startPosition;
    }

    public int getEndPosition() {
        return endPosition;
    }

    @Override
    public String toString() {
        return String.format("TextChunk(text=%s..., embedding_size=%d)", 
            text.substring(0, Math.min(50, text.length())), 
            embedding != null ? embedding.length : 0);
    }
} 
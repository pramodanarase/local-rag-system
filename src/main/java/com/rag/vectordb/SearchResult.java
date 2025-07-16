package com.rag.vectordb;

/**
 * Record representing a search result from the vector store.
 * @param id Unique identifier of the document
 * @param content The text content of the document
 * @param source The source file or location
 * @param score Similarity score (higher is more similar)
 */
public record SearchResult(String id, String content, String source, float score) {} 
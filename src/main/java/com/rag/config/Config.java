package com.rag.config;

import io.github.cdimascio.dotenv.Dotenv;

import java.nio.file.Path;
import java.util.List;
import java.util.Arrays;

public class Config {
    private static final Dotenv dotenv = Dotenv.load();

    // LM Studio settings
    public static final String LLAMA_HOST = dotenv.get("LLAMA_HOST", "localhost");
    public static final String LLAMA_PORT = dotenv.get("LLAMA_PORT", "1234");
    public static final String LLAMA_URL = String.format("http://%s:%s/v1", LLAMA_HOST, LLAMA_PORT);

    // Repository settings
    public static final String REPO_PATH = dotenv.get("REPO_PATH", "/Users/pa674029/code");
    public static final List<String> EXCLUDE_DIRS = Arrays.asList(
        ".git", "node_modules", "__pycache__", "venv", ".env", "target"
    );
    public static final List<String> INCLUDE_EXTENSIONS = Arrays.asList(
        ".java", ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".json", ".yaml", ".yml"
    );

    // Embedding model settings
    public static final String EMBEDDING_MODEL = "hf-internal-testing/tiny-random-bert";

    // Index settings
    public static final Path INDEX_PATH = Path.of("data", "index");
    public static final int CHUNK_SIZE = 1024;
    public static final int CHUNK_OVERLAP = 20;

    // Query settings
    public static final int MAX_TOKENS = 2000;
    public static final double TEMPERATURE = 0.7;
    public static final int TOP_K = 5;  // Number of relevant chunks to retrieve

    private Config() {
        // Private constructor to prevent instantiation
    }
} 
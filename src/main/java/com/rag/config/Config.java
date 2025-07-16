package com.rag.config;

import io.github.cdimascio.dotenv.Dotenv;

import java.nio.file.Path;
import java.util.List;
import java.util.Arrays;
import java.util.Optional;

public class Config {
    private static final Dotenv dotenv = Dotenv.load();

    // LM Studio settings
    public static final String LLAMA_HOST = getRequiredEnv("LLAMA_HOST");
    public static final String LLAMA_PORT = getRequiredEnv("LLAMA_PORT");
    public static final String LLAMA_URL = String.format("http://%s:%s/v1", LLAMA_HOST, LLAMA_PORT);

    // Repository settings
    public static final String REPO_PATH = getRequiredEnv("REPO_PATH");
    public static final List<String> EXCLUDE_DIRS = Arrays.asList(
        getEnv("EXCLUDE_DIRS", ".git,node_modules,__pycache__,venv,.env,target,test").split(",")
    );
    public static final List<String> INCLUDE_EXTENSIONS = Arrays.asList(
        getEnv("INCLUDE_EXTENSIONS", ".java").split(",")
    );

    // Index settings
    public static final Path INDEX_PATH = Path.of(getEnv("INDEX_PATH", "data/index"));
    public static final int CHUNK_SIZE = Integer.parseInt(getEnv("CHUNK_SIZE", "256"));
    public static final int CHUNK_OVERLAP = Integer.parseInt(getEnv("CHUNK_OVERLAP", "5"));

    // Query settings
    public static final int MAX_TOKENS = Integer.parseInt(getEnv("MAX_TOKENS", "500"));
    public static final double TEMPERATURE = Double.parseDouble(getEnv("TEMPERATURE", "0.7"));
    public static final int TOP_K = Integer.parseInt(getEnv("TOP_K", "3"));

    // Hugging Face configuration
    public static final String HF_TOKEN = getRequiredEnv("HF_TOKEN");
    public static final String EMBEDDING_MODEL = getEnv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2");
    public static final String LOCAL_MODEL_PATH = getEnv("LOCAL_MODEL_PATH", "models/all-MiniLM-L6-v2");
    public static final int EMBEDDING_DIMENSION = Integer.parseInt(getEnv("EMBEDDING_DIMENSION", "384"));

    private Config() {
        // Private constructor to prevent instantiation
    }

    /**
     * Gets an environment variable with a default value if not present
     */
    private static String getEnv(String key, String defaultValue) {
        return Optional.ofNullable(dotenv.get(key)).orElse(defaultValue);
    }

    /**
     * Gets a required environment variable, throws if not present
     */
    private static String getRequiredEnv(String key) {
        String value = dotenv.get(key);
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalStateException(
                String.format("Required environment variable '%s' is not set. Please check your .env file.", key)
            );
        }
        return value;
    }
} 
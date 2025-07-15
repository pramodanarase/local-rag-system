package com.rag.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.rag.config.Config;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class LMStudioClient implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(LMStudioClient.class);
    private static final MediaType JSON = MediaType.get("application/json");
    
    private final OkHttpClient client;
    private final ObjectMapper objectMapper;
    private final String apiUrl;

    public LMStudioClient() {
        this.client = new OkHttpClient.Builder()
            .connectTimeout(120, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(120, TimeUnit.SECONDS)
            .build();
        this.objectMapper = new ObjectMapper();
        this.apiUrl = Config.LLAMA_URL + "/chat/completions";
    }

    public String complete(String prompt, List<String> contextChunks) {
        try {
            // Format system message with context
            String systemMessage = formatSystemMessage(contextChunks);
            
            // Create the request body
            Map<String, Object> requestBody = Map.of(
                "messages", List.of(
                    Map.of("role", "system", "content", systemMessage),
                    Map.of("role", "user", "content", prompt)
                ),
                "temperature", Config.TEMPERATURE,
                "max_tokens", Config.MAX_TOKENS
            );

            // Convert request body to JSON
            String jsonBody = objectMapper.writeValueAsString(requestBody);

            // Create request
            Request request = new Request.Builder()
                .url(apiUrl)
                .post(RequestBody.create(jsonBody, JSON))
                .build();

            // Execute request
            try (Response response = client.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new RuntimeException("Unexpected response: " + response);
                }

                // Parse response
                String jsonResponse = response.body().string();
                Map<String, Object> responseMap = objectMapper.readValue(jsonResponse, Map.class);
                
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> choices = (List<Map<String, Object>>) responseMap.get("choices");
                if (choices == null || choices.isEmpty()) {
                    throw new RuntimeException("No choices in response");
                }

                @SuppressWarnings("unchecked")
                Map<String, String> message = (Map<String, String>) choices.get(0).get("message");
                return message.get("content");
            }
        } catch (Exception e) {
            logger.error("Error completing prompt", e);
            throw new RuntimeException("Failed to complete prompt", e);
        }
    }

    private String formatSystemMessage(List<String> contextChunks) {
        StringBuilder sb = new StringBuilder();
        sb.append("You are a helpful AI assistant that answers questions about code and documentation. ");
        sb.append("Use the following context to answer the user's question:\n\n");
        
        for (String chunk : contextChunks) {
            sb.append("---\n");
            sb.append(chunk);
            sb.append("\n");
        }
        
        sb.append("---\n\n");
        sb.append("If you cannot answer the question based on the context, say so. ");
        sb.append("Use code snippets in your answer when relevant.");
        
        return sb.toString();
    }

    @Override
    public void close() {
        if (client != null) {
            client.dispatcher().executorService().shutdown();
            client.connectionPool().evictAll();
        }
    }
} 
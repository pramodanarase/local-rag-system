package com.rag;

import com.rag.config.Config;
import com.rag.service.EmbeddingService;
import com.rag.service.LMStudioClient;
import com.rag.vectordb.LuceneVectorStore;
import com.rag.vectordb.SearchResult;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.StringJoiner;

public class QueryIndex {
    private static final Logger logger = LoggerFactory.getLogger(QueryIndex.class);
    private static final int MAX_HISTORY = 3; // Keep last 3 interactions for context

    private static class Interaction {
        final String query;
        final List<SearchResult> results;
        final String response;

        Interaction(String query, List<SearchResult> results, String response) {
            this.query = query;
            this.results = results;
            this.response = response;
        }
    }

    public static void main(String[] args) {
        try {
            // Initialize components
            Path indexPath = Config.INDEX_PATH;
            Directory directory = new NIOFSDirectory(indexPath);
            
            try (EmbeddingService embeddingService = new EmbeddingService();
                 LuceneVectorStore vectorStore = new LuceneVectorStore(directory);
                 LMStudioClient llmClient = new LMStudioClient();
                 Scanner scanner = new Scanner(System.in)) {
                
                QueryRepository queryRepository = new QueryRepository(embeddingService, vectorStore);
                List<Interaction> history = new ArrayList<>();

                System.out.println("RAG Query System");
                System.out.println("Type your query and press Enter. Type 'exit' to quit.");
                System.out.println("Type 'clear' to clear conversation history.");
                System.out.println("--------------------------------------------------");

                while (true) {
                    System.out.print("\nQuery> ");
                    String query = scanner.nextLine().trim();
                    
                    if (query.equalsIgnoreCase("exit")) {
                        break;
                    }
                    
                    if (query.equalsIgnoreCase("clear")) {
                        history.clear();
                        System.out.println("Conversation history cleared.");
                        continue;
                    }
                    
                    if (query.isEmpty()) {
                        continue;
                    }

                    try {
                        // Build context from history
                        String contextualQuery = buildContextualQuery(query, history);
                        
                        // Search for relevant documents
                        List<SearchResult> results = vectorStore.findSimilar(
                            embeddingService.generateEmbeddings(contextualQuery),
                            Config.TOP_K
                        );
                        
                        if (results.isEmpty()) {
                            System.out.println("No relevant documents found.");
                            continue;
                        }

                        // Build prompt for LLM
                        String prompt = buildPrompt(query, results, history);
                        // Get response from LLM
                        String response = llmClient.complete(prompt, new ArrayList<>());
                        
                        // Store interaction in history
                        history.add(new Interaction(query, results, response));
                        if (history.size() > MAX_HISTORY) {
                            history.remove(0);
                        }

                        // Display results
                        System.out.println("\nResponse:");
                        System.out.println("--------------------------------------------------");
                        System.out.println(response);
                        System.out.println("--------------------------------------------------");
                        
                        System.out.println("\nRelevant documents:");
                        System.out.println("--------------------------------------------------");
                        for (int i = 0; i < results.size(); i++) {
                            SearchResult result = results.get(i);
                            System.out.printf("%d. File: %s%n", i + 1, result.source());
                            System.out.printf("   Score: %.4f%n", result.score());
                            System.out.printf("   Content: %s%n", result.content());
                            System.out.println("--------------------------------------------------");
                        }
                    } catch (Exception e) {
                        logger.error("Error processing query: " + query, e);
                        System.out.println("Error: " + e.getMessage());
                    }
                }

                System.out.println("Goodbye!");
            }

        } catch (Exception e) {
            logger.error("Error initializing query system", e);
            System.exit(1);
        }
    }

    private static String buildContextualQuery(String query, List<Interaction> history) {
        if (history.isEmpty()) {
            return query;
        }

        StringJoiner joiner = new StringJoiner("\n");
        joiner.add(query);
        
        // Add context from previous interactions
        for (int i = history.size() - 1; i >= 0; i--) {
            Interaction interaction = history.get(i);
            joiner.add("Previous question: " + interaction.query);
            // Add most relevant snippet from previous results
            if (!interaction.results.isEmpty()) {
                joiner.add("Related context: " + interaction.results.get(0).content());
            }
        }

        return joiner.toString();
    }

    private static String buildPrompt(String query, List<SearchResult> results, List<Interaction> history) {
        StringJoiner joiner = new StringJoiner("\n\n");
        
        // Add system context
        joiner.add("You are a helpful assistant answering questions about a codebase. " +
                  "Use the provided context to answer questions accurately and concisely. " +
                  "If you're unsure or the context doesn't contain relevant information, say so.");

        // Add conversation history
        if (!history.isEmpty()) {
            joiner.add("Previous conversation:");
            for (Interaction interaction : history) {
                joiner.add("User: " + interaction.query);
                joiner.add("Assistant: " + interaction.response);
            }
        }

        // Add current context
        joiner.add("Context from codebase:");
        for (SearchResult result : results) {
            joiner.add("File: " + result.source() + "\n" + result.content());
        }

        // Add current query
        joiner.add("Current question: " + query);
        joiner.add("Answer the question based on the provided context. If referring to code, include the relevant file paths.");

        return joiner.toString();
    }
} 
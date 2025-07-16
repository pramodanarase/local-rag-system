package com.rag;

import com.rag.config.Config;
import com.rag.service.DocumentProcessor;
import com.rag.service.EmbeddingService;
import com.rag.vectordb.LuceneVectorStore;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.NIOFSDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class IndexFiles {
    private static final Logger logger = LoggerFactory.getLogger(IndexFiles.class);

    public static void main(String[] args) {
        // if (args.length < 1) {
        //     System.err.println("Usage: java -jar <jar-file> <directory-to-index>");
        //     System.exit(1);
        // }

        String directoryToIndex = "/Users/pa674029/code/Train-code";
        Path sourcePath = Paths.get(directoryToIndex);
        
        if (!Files.exists(sourcePath) || !Files.isDirectory(sourcePath)) {
            System.err.println("Error: " + directoryToIndex + " does not exist or is not a directory");
            System.exit(1);
        }

        Directory directory = null;
        try {
            // Initialize components
            Path indexPath = Config.INDEX_PATH;
            // Create index directory if it doesn't exist
            Files.createDirectories(indexPath);
            
            // Open directory and keep it open until the end
            directory = new NIOFSDirectory(indexPath);
            logger.info("Opened index directory at: {}", indexPath);
            
            try (EmbeddingService embeddingService = new EmbeddingService();
                 DocumentProcessor documentProcessor = new DocumentProcessor();
                 LuceneVectorStore vectorStore = new LuceneVectorStore(directory)) {
                
                IndexRepository indexRepository = new IndexRepository(documentProcessor, embeddingService, vectorStore);

                // Walk through directory and index files
                try (Stream<Path> paths = Files.walk(sourcePath)) {
                    paths.filter(path -> {
                            // Check if file should be processed
                            String fileName = path.getFileName().toString();
                            return Files.isRegularFile(path) &&
                                   Config.INCLUDE_EXTENSIONS.stream().anyMatch(ext -> fileName.endsWith(ext)) &&
                                   Config.EXCLUDE_DIRS.stream().noneMatch(dir -> path.toString().contains(dir));
                        })
                        .forEach(path -> {
                            try {
                                logger.info("Indexing file: {}", path);
                                indexRepository.indexFile(path);
                            } catch (Exception e) {
                                logger.error("Error indexing file: " + path, e);
                            }
                        });
                }

                // Final flush to ensure all documents are committed
                vectorStore.flush();
                logger.info("Indexing completed successfully!");
                logger.info("Files have been indexed to: {}", indexPath);
            }

        } catch (Exception e) {
            logger.error("Error during indexing", e);
            System.exit(1);
        } finally {
            // Close directory in finally block
            if (directory != null) {
                try {
                    directory.close();
                    logger.info("Closed index directory");
                } catch (IOException e) {
                    logger.error("Error closing index directory", e);
                }
            }
        }
    }
} 
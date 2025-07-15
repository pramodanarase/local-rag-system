package com.rag.service;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import com.rag.config.Config;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Paths;

public class EmbeddingService implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(EmbeddingService.class);
    private static final int MAX_SEQUENCE_LENGTH = 512;

    private final ZooModel<String, float[]> model;
    private final Predictor<String, float[]> predictor;
    private final HuggingFaceTokenizer tokenizer;

    public EmbeddingService() {
        try {
            // Initialize the tokenizer
            tokenizer = HuggingFaceTokenizer.newInstance(Config.EMBEDDING_MODEL);

            // Create model criteria
            Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelPath(Paths.get("models")) // Local model cache directory
                .optModelName(Config.EMBEDDING_MODEL)
                .optTranslator(new BertEmbeddingTranslator(tokenizer))
                .optProgress(new ProgressBar())
                .build();

            // Load the model
            model = criteria.loadModel();
            predictor = model.newPredictor();
            
            logger.info("Initialized embedding service with model: {}", Config.EMBEDDING_MODEL);
        } catch (Exception e) {
            logger.error("Failed to initialize embedding service", e);
            throw new RuntimeException("Failed to initialize embedding service", e);
        }
    }

    public float[] generateEmbedding(String text) {
        try {
            // Generate embeddings using the model
            float[] embedding = predictor.predict(text);
            
            // Normalize the embedding
            float sumSquares = 0;
            for (float value : embedding) {
                sumSquares += value * value;
            }
            float norm = (float) Math.sqrt(sumSquares);
            
            if (norm > 0) {
                for (int i = 0; i < embedding.length; i++) {
                    embedding[i] /= norm;
                }
            }
            
            logger.debug("Generated embedding of size: {}", embedding.length);
            return embedding;
        } catch (Exception e) {
            logger.error("Error generating embedding", e);
            throw new RuntimeException("Failed to generate embedding", e);
        }
    }

    @Override
    public void close() {
        try {
            if (predictor != null) {
                predictor.close();
            }
            if (model != null) {
                model.close();
            }
            if (tokenizer != null) {
                tokenizer.close();
            }
        } catch (Exception e) {
            logger.error("Error closing resources", e);
        }
    }

    private static class BertEmbeddingTranslator implements Translator<String, float[]> {
        private final HuggingFaceTokenizer tokenizer;

        public BertEmbeddingTranslator(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            // Get the CLS token embedding (first token)
            return list.get(0).toFloatArray();
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            // Tokenize and prepare input
            var encoding = tokenizer.encode(input);
            NDManager manager = ctx.getNDManager();
            
            long[] inputIds = encoding.getIds();
            long[] attentionMask = encoding.getAttentionMask();
            long[] tokenTypeIds = encoding.getTypeIds();
            
            // Truncate or pad to MAX_SEQUENCE_LENGTH
            int actualLength = Math.min(inputIds.length, MAX_SEQUENCE_LENGTH);
            Shape sequenceShape = new Shape(actualLength);
            
            NDArray inputIdsArray = manager.create(inputIds).reshape(sequenceShape);
            NDArray attentionMaskArray = manager.create(attentionMask).reshape(sequenceShape);
            NDArray tokenTypeIdsArray = manager.create(tokenTypeIds).reshape(sequenceShape);
            
            return new NDList(
                inputIdsArray.expandDims(0),
                attentionMaskArray.expandDims(0),
                tokenTypeIdsArray.expandDims(0)
            );
        }
    }
} 
package com.rag.service;

import ai.djl.ModelException;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import com.rag.config.Config;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class EmbeddingService implements Closeable {
    private static final Logger logger = LoggerFactory.getLogger(EmbeddingService.class);
    private static final int EMBEDDING_SIZE = 384;
    private static final int MAX_LENGTH = 512;

    private final ZooModel<String, float[]> model;
    private final Predictor<String, float[]> predictor;

    public EmbeddingService() throws IOException {
        try {
            // Load model from local path
            this.model = loadModel("sentence-transformers/all-MiniLM-L6-v2");
            this.predictor = model.newPredictor();
            logger.info("Model loaded successfully from sentence-transformers/all-MiniLM-L6-v2");
        } catch (ModelException e) {
            throw new IOException("Failed to load model: " + e.getMessage(), e);
        }
    }

    private static ZooModel<String, float[]> loadModel(String id) throws IOException, ModelException {
        return Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelUrls("djl://ai.djl.huggingface.pytorch/" + id)
                .optEngine("PyTorch")
                .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                .optProgress(new ProgressBar())
                .build()
                .loadModel();
    }

    public float[] generateEmbeddings(String text) throws TranslateException {
        return predictor.predict(text);
    }

    @Override
    public void close() throws IOException {
        predictor.close();
        model.close();
    }
}
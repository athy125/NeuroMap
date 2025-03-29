/**
 * GGML Integration Header for Model Activation Visualizer
 * 
 * This file contains helper functions and structures for integrating
 * with the GGML and llama.cpp libraries.
 */

 #ifndef GGML_INTEGRATION_H
 #define GGML_INTEGRATION_H
 
 #include "ggml.h"
 #include "ggml-alloc.h"
 #include "ggml-backend.h"
 #include "llama.h"
 
 #include <stddef.h>
 #include <stdbool.h>
 
 /**
  * GGML model context structure
  */
 typedef struct {
     struct llama_model* model;       // GGML model
     struct llama_context* ctx;       // GGML context
     
     struct ggml_allocr* allocator;   // GGML allocator
     struct ggml_backend* backend;    // GGML computation backend
     
     int n_layers;                    // Number of model layers
     int n_heads;                     // Number of attention heads
     int embedding_dim;               // Embedding dimension
     int vocab_size;                  // Vocabulary size
     int context_size;                // Maximum context size
     
     char* error_msg;                 // Error message buffer
     size_t error_msg_size;           // Size of error message buffer
 } GGMLModelContext;
 
 /**
  * Initialize GGML model from file
  * 
  * @param model_path Path to GGML model file
  * @param error_msg Buffer to store error message (can be NULL)
  * @param error_msg_size Size of error message buffer
  * @return Initialized GGML model context or NULL on error
  */
 GGMLModelContext* ggml_init_model(const char* model_path, char* error_msg, size_t error_msg_size);
 
 /**
  * Free GGML model resources
  * 
  * @param ctx GGML model context
  */
 void ggml_free_model_context(GGMLModelContext* ctx);
 
 /**
  * Tokenize text using GGML model
  * 
  * @param ctx GGML model context
  * @param text Text to tokenize
  * @param tokens Output buffer for tokens
  * @param n_tokens Size of tokens buffer (in) / number of tokens (out)
  * @param add_bos Whether to add beginning-of-sequence token
  * @return 0 on success, negative on error (with abs value = number of tokens that would be needed)
  */
 int ggml_tokenize(GGMLModelContext* ctx, const char* text, llama_token* tokens, int* n_tokens, bool add_bos);
 
 /**
  * Decode tokens to text
  * 
  * @param ctx GGML model context
  * @param tokens Tokens to decode
  * @param n_tokens Number of tokens
  * @return Decoded text (caller must free) or NULL on error
  */
 char* ggml_decode(GGMLModelContext* ctx, const llama_token* tokens, int n_tokens);
 
 /**
  * Get token text for a single token
  * 
  * @param ctx GGML model context
  * @param token Token to decode
  * @return Token text (managed by llama.cpp, do not free) or NULL on error
  */
 const char* ggml_token_to_str(GGMLModelContext* ctx, llama_token token);
 
 /**
  * Generate a single token with the model
  * 
  * @param ctx GGML model context
  * @param tokens Input token sequence
  * @param n_tokens Number of input tokens
  * @param temperature Sampling temperature (0 = greedy)
  * @param top_k Top-K sampling parameter (0 = disabled)
  * @param top_p Top-P sampling parameter (1.0 = disabled)
  * @param repeat_penalty Repetition penalty (1.0 = no penalty)
  * @param logits Output for token logits (vocab_size elements, can be NULL)
  * @param probs Output for token probabilities (vocab_size elements, can be NULL)
  * @return Generated token or -1 on error
  */
 llama_token ggml_generate_token(
     GGMLModelContext* ctx,
     const llama_token* tokens,
     int n_tokens,
     float temperature,
     int top_k,
     float top_p,
     float repeat_penalty,
     float* logits,
     float* probs
 );
 
 /**
  * Get layer activations from the model
  * 
  * @param ctx GGML model context
  * @param layer_idx Layer index
  * @param activation_type Type of activation to extract (0=MLP, 1=attention)
  * @return Layer activation value or NaN on error
  */
 float ggml_get_activation(GGMLModelContext* ctx, int layer_idx, int activation_type);
 
 /**
  * Structure to hold model state information
  */
 typedef struct {
     float* mlp_activations;          // MLP activations per layer
     float* attention_entropy;        // Attention entropy per layer
     float* logits;                   // Raw logits for all tokens
     float* probs;                    // Token probabilities
     llama_token* top_tokens;         // Top token IDs
     float* top_probs;                // Top token probabilities
     int n_top_tokens;                // Number of top tokens
 } GGMLModelState;
 
 /**
  * Initialize model state structure
  * 
  * @param ctx GGML model context
  * @param n_top_tokens Number of top tokens to track
  * @return Initialized model state or NULL on error
  */
 GGMLModelState* ggml_init_model_state(GGMLModelContext* ctx, int n_top_tokens);
 
 /**
  * Free model state resources
  * 
  * @param state Model state
  */
 void ggml_free_model_state(GGMLModelState* state);
 
 /**
  * Update model state with current generation info
  * 
  * @param ctx GGML model context
  * @param state Model state to update
  * @param tokens Input tokens
  * @param n_tokens Number of input tokens
  * @return 0 on success, negative on error
  */
 int ggml_update_model_state(
     GGMLModelContext* ctx,
     GGMLModelState* state,
     const llama_token* tokens,
     int n_tokens
 );
 
 #endif // GGML_INTEGRATION_H
/**
 * Model Activation Visualizer (MAV) - Implementation
 * 
 * A terminal-based visualization tool for transformer model activation patterns
 * during text generation.
 */

 #include "mav.h"
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 #include <time.h>
 #include <unistd.h>
 #include <ctype.h>
 #include <float.h>
 #include <stdarg.h>
 #include <termios.h>
 #include <fcntl.h>
 #include <errno.h>
 
 // Include GGML and llama.cpp dependencies
 #include "ggml.h"
 #include "ggml-alloc.h"
 #include "ggml-backend.h"
 #include "llama.h"
 
 // Version information
 #define MAV_VERSION "1.0.0"
 
 // Terminal utilities
 #define CLEAR_SCREEN  "\x1b[2J\x1b[H"
 #define HIDE_CURSOR   "\x1b[?25l"
 #define SHOW_CURSOR   "\x1b[?25h"
 
 // ANSI color codes for terminal output
 #define COLOR_RESET   "\x1b[0m"
 #define COLOR_RED     "\x1b[31m"
 #define COLOR_GREEN   "\x1b[32m"
 #define COLOR_YELLOW  "\x1b[33m"
 #define COLOR_BLUE    "\x1b[34m"
 #define COLOR_MAGENTA "\x1b[35m"
 #define COLOR_CYAN    "\x1b[36m"
 #define COLOR_WHITE   "\x1b[37m"
 #define COLOR_BOLD    "\x1b[1m"
 #define HIGHLIGHT     "\x1b[7m"
 
 // Safe memory operations
 #define SAFE_FREE(ptr) do { if (ptr) { free(ptr); (ptr) = NULL; } } while(0)
 #define CHECK_ALLOC(ptr, ctx) do { if (!(ptr)) { snprintf(ctx->error, sizeof(ctx->error), "Memory allocation failed at %s:%d", __FILE__, __LINE__); return NULL; } } while(0)
 
 // Token buffer for efficient token handling
 typedef struct {
     llama_token* tokens;
     int capacity;
     int count;
 } TokenBuffer;
 
 // Visualization data
 typedef struct {
     float mlp_activations[MAV_MAX_LAYERS];
     float attention_entropy[MAV_MAX_LAYERS];
     llama_token top_token_ids[MAV_TOP_K_PREDICTIONS];
     float top_token_probs[MAV_TOP_K_PREDICTIONS];
     float top_token_logits[MAV_TOP_K_PREDICTIONS];
     float* output_distribution;
     llama_token* generated_ids;
     int generated_length;
     llama_token next_token_id;
 } VisualizationData;
 
 // Main MAV context
 struct MavContext {
     // GGML/llama.cpp components
     struct llama_model* model;
     struct llama_context* ctx;
     
     // Tokenization and generation
     TokenBuffer tokens;
     VisualizationData viz_data;
     
     // Model information
     char model_name[MAV_MAX_PATH_LENGTH];
     int n_layers;
     int n_heads;
     int embedding_dim;
     int vocab_size;
     int context_size;
     
     // Parameters
     MavParams params;
     
     // Error handling
     char error[1024];
     bool has_error;
 };
 
 // Forward declarations of internal functions
 static int init_token_buffer(TokenBuffer* buffer, int initial_capacity);
 static void free_token_buffer(TokenBuffer* buffer);
 static int resize_token_buffer(TokenBuffer* buffer, int new_capacity);
 static int tokenize_text(MavContext* ctx, const char* text);
 static char* decode_tokens(MavContext* ctx, llama_token* tokens, int length);
 static int generate_next_token(MavContext* ctx);
 static void normalize_values(float* values, int length, MavScaleType scale_type, int max_bar_length);
 static void render_visualization(MavContext* ctx);
 static void set_error(MavContext* ctx, const char* format, ...);
 static void render_mlp_panel(MavContext* ctx, int max_bar_length);
 static void render_entropy_panel(MavContext* ctx, int max_bar_length);
 static void render_top_predictions_panel(MavContext* ctx);
 static void render_probability_dist_panel(MavContext* ctx, int max_bar_length);
 static void render_text_panel(MavContext* ctx, int limit_chars);
 
 /**
  * Initialize a token buffer
  */
 static int init_token_buffer(TokenBuffer* buffer, int initial_capacity) {
     if (buffer == NULL || initial_capacity <= 0) {
         return MAV_ERR_INVALID_PARAM;
     }
     
     buffer->tokens = (llama_token*)malloc(sizeof(llama_token) * initial_capacity);
     if (buffer->tokens == NULL) {
         return MAV_ERR_MEMORY;
     }
     
     buffer->capacity = initial_capacity;
     buffer->count = 0;
     
     return MAV_SUCCESS;
 }
 
 /**
  * Free a token buffer
  */
 static void free_token_buffer(TokenBuffer* buffer) {
     if (buffer == NULL) {
         return;
     }
     
     SAFE_FREE(buffer->tokens);
     buffer->capacity = 0;
     buffer->count = 0;
 }
 
 /**
  * Resize a token buffer
  */
 static int resize_token_buffer(TokenBuffer* buffer, int new_capacity) {
     if (buffer == NULL || new_capacity <= 0 || new_capacity < buffer->count) {
         return MAV_ERR_INVALID_PARAM;
     }
     
     llama_token* new_tokens = (llama_token*)realloc(buffer->tokens, sizeof(llama_token) * new_capacity);
     if (new_tokens == NULL) {
         return MAV_ERR_MEMORY;
     }
     
     buffer->tokens = new_tokens;
     buffer->capacity = new_capacity;
     
     return MAV_SUCCESS;
 }
 
 /**
  * Set error message in MAV context
  */
 static void set_error(MavContext* ctx, const char* format, ...) {
     if (ctx == NULL) {
         return;
     }
     
     va_list args;
     va_start(args, format);
     
     ctx->has_error = true;
     vsnprintf(ctx->error, sizeof(ctx->error), format, args);
     
     // Print the error to stderr in verbose mode
     if (ctx->params.verbose) {
         fprintf(stderr, "MAV Error: ");
         vfprintf(stderr, format, args);
         fprintf(stderr, "\n");
     }
     
     va_end(args);
 }
 
 /**
  * Normalize values for visualization
  */
 static void normalize_values(float* values, int length, MavScaleType scale_type, int max_bar_length) {
     if (values == NULL || length <= 0 || max_bar_length <= 0) {
         return;
     }
     
     float min_val = FLT_MAX;
     float max_val = -FLT_MAX;
     
     // Find min/max
     for (int i = 0; i < length; i++) {
         if (values[i] < min_val) min_val = values[i];
         if (values[i] > max_val) max_val = values[i];
     }
     
     // Apply scaling transformation
     if (scale_type == MAV_SCALE_LOG) {
         for (int i = 0; i < length; i++) {
             values[i] = log1pf(fabsf(values[i]));
         }
         
         // Recalculate min/max
         min_val = FLT_MAX;
         max_val = -FLT_MAX;
         for (int i = 0; i < length; i++) {
             if (values[i] < min_val) min_val = values[i];
             if (values[i] > max_val) max_val = values[i];
         }
     } else if (scale_type == MAV_SCALE_MINMAX) {
         if (max_val - min_val > 1e-9) {
             for (int i = 0; i < length; i++) {
                 values[i] = (values[i] - min_val) / (max_val - min_val);
             }
         } else {
             for (int i = 0; i < length; i++) {
                 values[i] = 0;
             }
         }
     }
     
     // Scale to max_bar_length
     if (max_val > 0) {
         float scale_factor = (float)max_bar_length / max_val;
         for (int i = 0; i < length; i++) {
             values[i] *= scale_factor;
         }
     }
 }
 
 /**
  * Tokenize input text
  */
 static int tokenize_text(MavContext* ctx, const char* text) {
     if (ctx == NULL || text == NULL) {
         return MAV_ERR_INVALID_PARAM;
     }
     
     // Estimate token capacity (usually 1.5x characters is sufficient)
     int estimated_tokens = strlen(text) * 1.5;
     if (estimated_tokens < 32) estimated_tokens = 32;
     
     // Ensure token buffer has enough capacity
     if (ctx->tokens.capacity < estimated_tokens) {
         int result = resize_token_buffer(&ctx->tokens, estimated_tokens);
         if (result != MAV_SUCCESS) {
             set_error(ctx, "Failed to resize token buffer");
             return result;
         }
     }
     
     // Tokenize the text
     int n_tokens = llama_tokenize(ctx->model, text, strlen(text), 
                                  ctx->tokens.tokens, ctx->tokens.capacity, true, false);
     
     if (n_tokens < 0) {
         n_tokens = -n_tokens;
         set_error(ctx, "Input too long, truncated to %d tokens", n_tokens);
         return MAV_ERR_TOKENIZATION;
     }
     
     ctx->tokens.count = n_tokens;
     
     // Copy tokens to generated IDs
     if (ctx->viz_data.generated_ids == NULL || 
         ctx->viz_data.generated_length + n_tokens > MAV_MAX_TOKENS) {
         SAFE_FREE(ctx->viz_data.generated_ids);
         ctx->viz_data.generated_ids = (llama_token*)malloc(sizeof(llama_token) * MAV_MAX_TOKENS);
         if (ctx->viz_data.generated_ids == NULL) {
             set_error(ctx, "Failed to allocate memory for generated tokens");
             return MAV_ERR_MEMORY;
         }
         ctx->viz_data.generated_length = 0;
     }
     
     memcpy(ctx->viz_data.generated_ids, ctx->tokens.tokens, n_tokens * sizeof(llama_token));
     ctx->viz_data.generated_length = n_tokens;
     
     return MAV_SUCCESS;
 }
 
 /**
  * Decode tokens to text
  */
 static char* decode_tokens(MavContext* ctx, llama_token* tokens, int length) {
     if (ctx == NULL || tokens == NULL || length <= 0) {
         return NULL;
     }
     
     // Allocate a buffer for the decoded text
     // Estimate initial size (typically 4 bytes per token is sufficient)
     size_t buffer_size = length * 8;
     char* text = (char*)malloc(buffer_size);
     if (text == NULL) {
         set_error(ctx, "Failed to allocate memory for decoded text");
         return NULL;
     }
     
     text[0] = '\0';
     size_t total_len = 0;
     
     // Decode each token
     for (int i = 0; i < length; i++) {
         llama_token id = tokens[i];
         
         if (id < 0 || id >= ctx->vocab_size) {
             // Skip invalid tokens
             continue;
         }
         
         // Get token text
         const char* token_text = llama_token_to_piece(ctx->model, id);
         if (token_text == NULL) {
             continue;
         }
         
         size_t token_len = strlen(token_text);
         
         // Check if we need to grow the buffer
         if (total_len + token_len + 1 > buffer_size) {
             buffer_size *= 2;
             char* new_text = (char*)realloc(text, buffer_size);
             if (new_text == NULL) {
                 set_error(ctx, "Failed to reallocate memory for decoded text");
                 free(text);
                 return NULL;
             }
             text = new_text;
         }
         
         // Copy this token's text to the buffer
         memcpy(text + total_len, token_text, token_len);
         total_len += token_len;
         text[total_len] = '\0';
     }
     
     return text;
 }
 
 /**
  * Generate the next token
  */
 static int generate_next_token(MavContext* ctx) {
     if (ctx == NULL) {
         return MAV_ERR_INVALID_PARAM;
     }
     
     if (ctx->viz_data.generated_length <= 0) {
         set_error(ctx, "No tokens in the sequence");
         return MAV_ERR_INVALID_PARAM;
     }
     
     // Create input batch for the model
     struct llama_batch batch = llama_batch_init(
         ctx->viz_data.generated_length, 0, 1
     );
     
     // Fill the batch with our tokens
     for (int i = 0; i < ctx->viz_data.generated_length; i++) {
         llama_batch_add(&batch, ctx->viz_data.generated_ids[i], i, { 0 }, false);
     }
     
     // Run the model forward pass
     if (llama_decode(ctx->ctx, batch) != 0) {
         llama_batch_free(batch);
         set_error(ctx, "Failed to run model forward pass");
         return MAV_ERR_GENERATION;
     }
     
     // Get logits for the next token
     float* logits = llama_get_logits(ctx->ctx);
     
     // Get hidden states (for MLP activations)
     const int n_layers = ctx->n_layers;
     
     // Extract layer activations
     for (int i = 0; i < n_layers && i < MAV_MAX_LAYERS; i++) {
         // Get MLP activations (using norm values as proxy)
         float activation = 0.0f;
         const float* layer_output = NULL;
         
         // In real implementation, you would use llama API to get activations:
         // layer_output = llama_get_layer_output(ctx->ctx, i);
         
         // Placeholder - in real impl, calculate L2 norm of activations
         ctx->viz_data.mlp_activations[i] = (float)(i + 1) / n_layers * 5.0f;
         
         // Get attention entropy (placeholder in this example)
         ctx->viz_data.attention_entropy[i] = (float)(n_layers - i) / n_layers * 3.0f;
     }
     
     // Prepare token data array for sampling
     llama_token_data_array candidates = {
         .data = (llama_token_data*)malloc(ctx->vocab_size * sizeof(llama_token_data)),
         .size = ctx->vocab_size,
         .sorted = false
     };
     
     if (candidates.data == NULL) {
         llama_batch_free(batch);
         set_error(ctx, "Failed to allocate memory for token candidates");
         return MAV_ERR_MEMORY;
     }
     
     // Fill token data array with logits
     for (int token_id = 0; token_id < ctx->vocab_size; token_id++) {
         candidates.data[token_id].id = token_id;
         candidates.data[token_id].logit = logits[token_id];
         candidates.data[token_id].p = 0.0f;
     }
     
     // Apply sampling parameters
     if (ctx->params.temperature <= 0.0f) {
         // Greedy sampling
         llama_sample_softmax(ctx->ctx, &candidates);
         ctx->viz_data.next_token_id = candidates.data[0].id;
     } else {
         // Temperature sampling with various parameters
         llama_sample_repetition_penalty(ctx->ctx, &candidates,
                                       ctx->viz_data.generated_ids,
                                       ctx->viz_data.generated_length,
                                       ctx->params.repetition_penalty);
         
         llama_sample_top_k(ctx->ctx, &candidates, ctx->params.top_k, 1);
         llama_sample_top_p(ctx->ctx, &candidates, ctx->params.top_p, 1);
         llama_sample_temperature(ctx->ctx, &candidates, ctx->params.temperature);
         
         ctx->viz_data.next_token_id = llama_sample_token(ctx->ctx, &candidates);
     }
     
     // Ensure we have output distribution for visualization
     if (ctx->viz_data.output_distribution == NULL) {
         ctx->viz_data.output_distribution = (float*)malloc(ctx->vocab_size * sizeof(float));
         if (ctx->viz_data.output_distribution == NULL) {
             free(candidates.data);
             llama_batch_free(batch);
             set_error(ctx, "Failed to allocate memory for output distribution");
             return MAV_ERR_MEMORY;
         }
     }
     
     // Copy probabilities for visualization
     llama_sample_softmax(ctx->ctx, &candidates);
     for (int i = 0; i < ctx->vocab_size; i++) {
         // Default to zero
         ctx->viz_data.output_distribution[i] = 0.0f;
     }
     
     for (int i = 0; i < candidates.size; i++) {
         llama_token token_id = candidates.data[i].id;
         float prob = candidates.data[i].p;
         if (token_id >= 0 && token_id < ctx->vocab_size) {
             ctx->viz_data.output_distribution[token_id] = prob;
         }
     }
     
     // Store top-k predictions for display
     llama_sample_softmax(ctx->ctx, &candidates);
     int top_k = MAV_TOP_K_PREDICTIONS < candidates.size ? 
                MAV_TOP_K_PREDICTIONS : candidates.size;
     
     for (int i = 0; i < top_k; i++) {
         ctx->viz_data.top_token_ids[i] = candidates.data[i].id;
         ctx->viz_data.top_token_probs[i] = candidates.data[i].p;
         ctx->viz_data.top_token_logits[i] = candidates.data[i].logit;
     }
     
     // Cleanup
     free(candidates.data);
     llama_batch_free(batch);
     
     return MAV_SUCCESS;
 }
 
 /**
  * Render MLP activations panel
  */
 static void render_mlp_panel(MavContext* ctx, int max_bar_length) {
     if (ctx == NULL || max_bar_length <= 0) {
         return;
     }
     
     printf("\n%s=== MLP Activations ===%s\n", COLOR_CYAN, COLOR_RESET);
     
     // Safety check for number of layers
     int n_layers = ctx->n_layers;
     if (n_layers <= 0 || n_layers > MAV_MAX_LAYERS) {
         printf("%sNo valid layer information available%s\n", COLOR_RED, COLOR_RESET);
         return;
     }
     
     float normalized[MAV_MAX_LAYERS];
     for (int i = 0; i < n_layers; i++) {
         normalized[i] = ctx->viz_data.mlp_activations[i];
     }
     
     normalize_values(normalized, n_layers, ctx->params.scale_type, max_bar_length);
     
     for (int i = 0; i < n_layers; i++) {
         int bar_length = (int)normalized[i];
         
         // Bound checking for bar length
         if (bar_length < 0) bar_length = 0;
         if (bar_length > max_bar_length) bar_length = max_bar_length;
         
         printf("%sLayer %2d%s | %s:%s ", COLOR_BOLD, i, COLOR_RESET, COLOR_YELLOW, COLOR_RESET);
         
         for (int j = 0; j < bar_length; j++) {
             printf("█");
         }
         for (int j = bar_length; j < max_bar_length; j++) {
             printf(" ");
         }
         
         printf(" %s%+.3f%s\n", COLOR_YELLOW, ctx->viz_data.mlp_activations[i], COLOR_RESET);
     }
 }
 
 /**
  * Render attention entropy panel
  */
 static void render_entropy_panel(MavContext* ctx, int max_bar_length) {
     if (ctx == NULL || max_bar_length <= 0) {
         return;
     }
     
     printf("\n%s=== Attention Entropy ===%s\n", COLOR_MAGENTA, COLOR_RESET);
     
     // Safety check for number of layers
     int n_layers = ctx->n_layers;
     if (n_layers <= 0 || n_layers > MAV_MAX_LAYERS) {
         printf("%sNo valid layer information available%s\n", COLOR_RED, COLOR_RESET);
         return;
     }
     
     float normalized[MAV_MAX_LAYERS];
     for (int i = 0; i < n_layers; i++) {
         normalized[i] = ctx->viz_data.attention_entropy[i];
     }
     
     normalize_values(normalized, n_layers, ctx->params.scale_type, max_bar_length);
     
     for (int i = 0; i < n_layers; i++) {
         int bar_length = (int)normalized[i];
         
         // Bound checking for bar length
         if (bar_length < 0) bar_length = 0;
         if (bar_length > max_bar_length) bar_length = max_bar_length;
         
         printf("%sLayer %2d%s | %s:%s ", COLOR_BOLD, i, COLOR_RESET, COLOR_YELLOW, COLOR_RESET);
         
         for (int j = 0; j < bar_length; j++) {
             printf("█");
         }
         for (int j = bar_length; j < max_bar_length; j++) {
             printf(" ");
         }
         
         printf(" %.3f\n", ctx->viz_data.attention_entropy[i]);
     }
 }
 
 /**
  * Render top predictions panel
  */
 static void render_top_predictions_panel(MavContext* ctx) {
     if (ctx == NULL) {
         return;
     }
     
     printf("\n%s=== Top Predictions ===%s\n", COLOR_BLUE, COLOR_RESET);
     
     // Check if we have valid top predictions
     bool has_valid_predictions = false;
     for (int i = 0; i < MAV_TOP_K_PREDICTIONS; i++) {
         if (ctx->viz_data.top_token_ids[i] >= 0 && 
             ctx->viz_data.top_token_ids[i] < ctx->vocab_size) {
             has_valid_predictions = true;
             break;
         }
     }
     
     if (!has_valid_predictions) {
         printf("%sNo valid predictions available%s\n", COLOR_RED, COLOR_RESET);
         return;
     }
     
     for (int row = 0; row < 4; row++) {
         for (int col = 0; col < 5; col++) {
             int idx = row * 5 + col;
             if (idx < MAV_TOP_K_PREDICTIONS) {
                 int token_id = ctx->viz_data.top_token_ids[idx];
                 
                 // Validate token ID
                 if (token_id < 0 || token_id >= ctx->vocab_size) {
                     printf("%s%-10s%s ", COLOR_RED, "<?INVALID>", COLOR_RESET);
                     continue;
                 }
                 
                 const char* token = llama_token_to_piece(ctx->model, token_id);
                 if (token == NULL) {
                     printf("%s%-10s%s ", COLOR_RED, "<?NULL>", COLOR_RESET);
                     continue;
                 }
                 
                 // Truncate token display if needed
                 char display_token[11] = {0}; // 10 chars + null terminator
                 strncpy(display_token, token, 10);
                 
                 // Replace control characters and ensure proper display
                 for (int i = 0; display_token[i] != '\0'; i++) {
                     if (iscntrl((unsigned char)display_token[i]) || 
                         !isprint((unsigned char)display_token[i])) {
                         display_token[i] = '?';
                     }
                 }
                 
                 // Format and print
                 printf("%s%-10s%s ", COLOR_MAGENTA, display_token, COLOR_RESET);
                 printf("(%s%5.1f%%%s, %s%4.1f%s)    ", 
                        COLOR_YELLOW, ctx->viz_data.top_token_probs[idx] * 100.0, COLOR_RESET,
                        COLOR_CYAN, ctx->viz_data.top_token_logits[idx], COLOR_RESET);
             }
         }
         printf("\n");
     }
 }
 
 /**
  * Render probability distribution panel
  */
 static void render_probability_dist_panel(MavContext* ctx, int max_bar_length) {
     if (ctx == NULL || max_bar_length <= 0) {
         return;
     }
     
     printf("\n%s=== Output Distribution ===%s\n", COLOR_YELLOW, COLOR_RESET);
     
     // Check if we have a valid distribution
     if (ctx->viz_data.output_distribution == NULL) {
         printf("%sNo output distribution available%s\n", COLOR_RED, COLOR_RESET);
         return;
     }
     
// For simplicity, we'll just bin the top 100 probabilities
const int num_bins = 10;
const int num_tokens = 100;

// Check vocabulary size
if (ctx->vocab_size <= 0) {
    printf("%sInvalid vocabulary size%s\n", COLOR_RED, COLOR_RESET);
    return;
}

// Sort probs
float sorted_probs[num_tokens];
for (int i = 0; i < num_tokens; i++) {
    sorted_probs[i] = 0.0f;
}

// Safe iteration over vocab
const int vocab_size = ctx->vocab_size < MAV_MAX_VOCAB_SIZE ? 
                      ctx->vocab_size : MAV_MAX_VOCAB_SIZE;

for (int i = 0; i < vocab_size; i++) {
    float prob = ctx->viz_data.output_distribution[i];
    
    if (!isfinite(prob)) {
        // Skip NaN or inf values
        continue;
    }
    
    // Insert into sorted array
    for (int j = 0; j < num_tokens; j++) {
        if (prob > sorted_probs[j]) {
            // Shift everything down
            for (int k = num_tokens - 1; k > j; k--) {
                sorted_probs[k] = sorted_probs[k-1];
            }
            sorted_probs[j] = prob;
            break;
        }
    }
}

// Create bins
float bin_sums[num_bins];
for (int i = 0; i < num_bins; i++) {
    bin_sums[i] = 0.0f;
}

int tokens_per_bin = num_tokens / num_bins;
if (tokens_per_bin < 1) tokens_per_bin = 1;

for (int i = 0; i < num_tokens; i++) {
    int bin = i / tokens_per_bin;
    if (bin < num_bins) {
        bin_sums[bin] += sorted_probs[i];
    }
}

// Find max sum
float max_sum = 0.0f;
for (int i = 0; i < num_bins; i++) {
    if (bin_sums[i] > max_sum) {
        max_sum = bin_sums[i];
    }
}

// Display bars
for (int i = 0; i < num_bins; i++) {
    int bin_edge = (i + 1) * tokens_per_bin - 1;
    if (bin_edge >= num_tokens) bin_edge = num_tokens - 1;
    
    float prob_value = sorted_probs[bin_edge];
    
    // Calculate bar length with bounds checking
    int bar_length = 0;
    if (max_sum > 0.0f) {
        bar_length = (int)((bin_sums[i] / max_sum) * max_bar_length);
        if (bar_length < 0) bar_length = 0;
        if (bar_length > max_bar_length) bar_length = max_bar_length;
    }
    
    printf("%s%.4f%s: %s", COLOR_YELLOW, prob_value, COLOR_RESET, COLOR_CYAN);
    for (int j = 0; j < bar_length; j++) {
        printf("█");
    }
    printf("%s\n", COLOR_RESET);
}
}
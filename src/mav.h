/**
 * Model Activation Visualizer (MAV) - Header
 * 
 * A terminal-based visualization tool for transformer model activation patterns
 * during text generation.
 */

 #ifndef MAV_H
 #define MAV_H
 
 #include <stdbool.h>
 #include <stdint.h>
 
 // Error codes
 #define MAV_SUCCESS 0
 #define MAV_ERR_MEMORY 1
 #define MAV_ERR_FILE_NOT_FOUND 2
 #define MAV_ERR_MODEL_LOAD 3
 #define MAV_ERR_TOKENIZATION 4
 #define MAV_ERR_GENERATION 5
 #define MAV_ERR_INVALID_PARAM 6
 
 // Model constants
 #define MAV_MAX_LAYERS 128
 #define MAV_MAX_TOKENS 4096
 #define MAV_MAX_VOCAB_SIZE 100000
 #define MAV_MAX_BAR_LENGTH 20
 #define MAV_MAX_PATH_LENGTH 512
 #define MAV_MAX_PROMPT_LENGTH 2048
 #define MAV_TOP_K_PREDICTIONS 20
 
 // Visualization scaling types
 typedef enum {
     MAV_SCALE_LINEAR,   // Linear scaling
     MAV_SCALE_LOG,      // Logarithmic scaling (better for wide ranges)
     MAV_SCALE_MINMAX    // Min-max normalization
 } MavScaleType;
 
 // Generation parameters
 typedef struct {
     float temperature;          // Sampling temperature (0.0 = greedy)
     int top_k;                  // Top-K sampling parameter
     float top_p;                // Top-P (nucleus) sampling
     float min_p;                // Min-P sampling parameter
     float repetition_penalty;   // Penalty for repeating tokens
     int max_new_tokens;         // Maximum tokens to generate
     float refresh_rate;         // Refresh rate in seconds
     bool interactive;           // Interactive mode flag
     MavScaleType scale_type;    // Visualization scaling
     int limit_chars;            // Limit for displayed chars
     bool verbose;               // Verbose output
     int seed;                   // Random seed
 } MavParams;
 
 // Main MAV context structure (opaque pointer)
 typedef struct MavContext MavContext;
 
 /**
  * Initialize MAV with a model path
  * 
  * @param model_path Path to the GGML model file
  * @param params Generation parameters
  * @param error_msg Buffer to store error message (can be NULL)
  * @param error_msg_size Size of error_msg buffer
  * @return MAV context or NULL on error
  */
 MavContext* mav_init(const char* model_path, MavParams* params, 
                      char* error_msg, size_t error_msg_size);
 
 /**
  * Free MAV context and resources
  * 
  * @param ctx MAV context
  */
 void mav_free(MavContext* ctx);
 
 /**
  * Set prompt for generation
  * 
  * @param ctx MAV context
  * @param prompt Text prompt to start generation
  * @param error_msg Buffer to store error message (can be NULL)
  * @param error_msg_size Size of error_msg buffer
  * @return 0 on success, error code on failure
  */
 int mav_set_prompt(MavContext* ctx, const char* prompt, 
                   char* error_msg, size_t error_msg_size);
 
 /**
  * Run generation with visualization
  * 
  * @param ctx MAV context
  * @param error_msg Buffer to store error message (can be NULL)
  * @param error_msg_size Size of error_msg buffer
  * @return 0 on success, error code on failure
  */
 int mav_run(MavContext* ctx, char* error_msg, size_t error_msg_size);
 
 /**
  * Get default parameters
  * 
  * @return Default MAV parameters
  */
 MavParams mav_default_params(void);
 
 /**
  * Get version string
  * 
  * @return Version string
  */
 const char* mav_version(void);
 
 #endif // MAV_H
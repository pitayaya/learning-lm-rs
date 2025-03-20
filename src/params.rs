use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use safetensors::Dtype;

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        safetensor.names().iter().for_each(|name| {
            println!("{}", name);
        });
        let get_tensor = |name: &str| -> Tensor<f32> {
            match safetensor.tensor(name) {
                Ok(tensor_view) => {
                    if tensor_view.dtype() != Dtype::F32 {
                        panic!("Unsupported dtype for tensor {}: {:?}", name, tensor_view.dtype());
                    }

                    let data_len = tensor_view.data().len() / std::mem::size_of::<f32>();
                    let data_array: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            tensor_view.data().as_ptr() as *const f32,
                            data_len,
                        )
                    };

                    Tensor::new(data_array.to_vec(), &tensor_view.shape().to_vec())
                },
                Err(e) => panic!("Failed to load tensor {}: {:?}", name, e),
            }
        };
        
        let embedding_table =  if config.tie_word_embeddings {
            get_tensor("lm_head.weight")
        } else {
            get_tensor("model.embed_tokens.weight")
        };

        let mut rms_att_w = Vec::with_capacity(config.num_hidden_layers);
        let mut wq = Vec::with_capacity(config.num_hidden_layers);
        let mut wk = Vec::with_capacity(config.num_hidden_layers);
        let mut wv = Vec::with_capacity(config.num_hidden_layers);
        let mut wo = Vec::with_capacity(config.num_hidden_layers);
        let mut rms_ffn_w = Vec::with_capacity(config.num_hidden_layers);
        let mut w_up = Vec::with_capacity(config.num_hidden_layers);
        let mut w_gate = Vec::with_capacity(config.num_hidden_layers);
        let mut w_down = Vec::with_capacity(config.num_hidden_layers);

        for i in 0..config.num_hidden_layers {
            rms_att_w.push(get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)));
            wq.push(get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)));
            wk.push(get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)));
            wv.push(get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)));
            wo.push(get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)));
            rms_ffn_w.push(get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)));
            w_up.push(get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)));
            w_gate.push(get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)));
            w_down.push(get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)));
        }

        let rms_out_w = get_tensor("model.norm.weight");
        let lm_head = get_tensor("lm_head.weight");

        LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        }
    }
}

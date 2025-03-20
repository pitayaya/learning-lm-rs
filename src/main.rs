mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::io::{self, Write};
use serde::{Deserialize, Serialize};
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use sqlx::postgres::PgPool;
use dotenv::dotenv;
use std::env;
use sqlx::postgres::PgPoolOptions;
use tera::Tera;
use model::Llama;

// fn main() {
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     let model_dir = PathBuf::from(project_dir).join("models").join("story");
//     let llama = model::Llama::<f32>::from_safetensors(&model_dir);
//     let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
//     let input = "one plus one equal to";
//     let binding = tokenizer.encode(input, true).unwrap();
//     let input_ids = binding.get_ids();
//     // print!("\n{}", input);
//     let output_ids = llama.generate(
//         input_ids,
//         500,
//         0.8,
//         30,
//         1.,
//     );
//     println!("{}", tokenizer.decode(&output_ids, true).unwrap());
// }
#[derive(Deserialize, Serialize, Debug)]
pub struct UserInput {
    pub chat_id: i32,
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct UserDelete {
    pub chat_id: i32,
}

#[derive(Serialize)]
struct ResponseMessage {
    response: String,
}

struct AppState {
    llama: Llama<f32>,
    tokenizer: Tokenizer,
    db: PgPool,
}

fn format_chat_template(messages: Vec<(&str, &String)>, add_generation_prompt: bool) -> String {
    let mut formatted = String::new();
    for (role, content) in messages {
        formatted.push_str(&format!("<|im_start|>{}", role));
        formatted.push('\n');
        formatted.push_str(content);
        formatted.push_str("<|im_end|>");
        formatted.push('\n');
    }
    if add_generation_prompt {
        formatted.push_str("<|im_start|>assistant\n");
    }
    formatted
}

fn clean_output(output: String) -> String {
    match output.find("assistant") {
        Some(index) => output[index + "assistant".len()..].trim().to_string(),
        None => output.trim().to_string(), 
    }
}

async fn save_text_to_db(pool: &PgPool, chat_id: i32, text: String) -> Result<(), sqlx::Error> {
    sqlx::query!(
        "INSERT INTO users (chat_id, text) VALUES ($1, $2)",
        chat_id,
        text
    )
    .execute(pool)
    .await?;
    Ok(())
}

async fn get_texts_for_user_db(pool: &PgPool, chat_id: i32) -> Result<Vec<UserInput>, sqlx::Error> {
    sqlx::query_as!(
        UserInput,
        "SELECT chat_id, text FROM users WHERE chat_id = $1",
        chat_id
    )
    .fetch_all(pool)
    .await
}

async fn delete_user_from_db(pool: &PgPool, chat_id: i32) -> Result<(), sqlx::Error> {
    sqlx::query!(
        "DELETE FROM users WHERE chat_id = $1",
        chat_id
    )
    .execute(pool)
    .await?;
    Ok(())
}

async fn send_message(
    user_input: web::Json<UserInput>,
    app_state: web::Data<AppState>,
) -> impl Responder {
    let user_input = user_input.into_inner();

    println!("begin to save to db");
    // 保存用户输入到数据库
    if let Err(e) = save_text_to_db(&app_state.db, user_input.chat_id, user_input.text.clone()).await {
        eprintln!("Database error: {}", e);
        return HttpResponse::InternalServerError().json(ResponseMessage { response: "Database error".to_string() });
    }

    println!("getting all texts");
    // 获取用户的所有文本
    let texts = match get_texts_for_user_db(&app_state.db, user_input.chat_id).await {
        Ok(texts) => texts,
        Err(e) => {
            eprintln!("Database error: {}", e);
            return HttpResponse::InternalServerError().json(ResponseMessage { response: "Database error".to_string() });
        }
    };

    // 使用所有文本构建对话模板
    let messages = texts.iter().map(|text| ("user", &text.text)).collect::<Vec<_>>();
    let formatted_input = format_chat_template(messages, true);

    // 编码并生成回复
    let encoded = match app_state.tokenizer.encode(formatted_input, true) {
        Ok(encoded) => encoded,
        Err(e) => {
            eprintln!("Tokenization error: {}", e);
            return HttpResponse::InternalServerError().body("Tokenization error");
        }
    };
    let input_ids = encoded.get_ids();

    let output_ids = app_state.llama.generate(
        input_ids,
        200,  // max_length
        0.8,  // top_p
        50,   // top_k
        1.0,  // temperature
    );

    let output = match app_state.tokenizer.decode(&output_ids, true) {
        Ok(output) => output,
        Err(e) => {
            eprintln!("Decoding error: {}", e);
            return HttpResponse::InternalServerError().body("Decoding error");
        }
    };

    let cleaned_output = clean_output(output);

    println!("生成回复");
    HttpResponse::Ok().json(ResponseMessage { response: cleaned_output })
}

async fn delete_send_reply(
    user_input: web::Json<UserDelete>,
    app_state: web::Data<AppState>,
) -> impl Responder {
    let user_input = user_input.into_inner();
    delete_user_from_db(&app_state.db, user_input.chat_id).await;
    HttpResponse::Ok()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL is not set.");
    let db_pool = PgPoolOptions::new().connect(&database_url).await.unwrap();

    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");

    let llama = Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let tera = Tera::new(concat!(env!("CARGO_MANIFEST_DIR"), "/templates/**/*")).unwrap();

    let app_state = web::Data::new(AppState {
        llama,
        tokenizer,
        db: db_pool,
    });

    println!("生成页面");
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .app_data(web::Data::new(tera.clone()))
            .route("/", web::get().to(chat))
            .service(web::resource("/1").route(web::post().to(send_message)))
            .service(web::resource("/2").route(web::post().to(delete_send_reply)))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

async fn chat(tera: web::Data<Tera>) -> Result<HttpResponse, actix_web::Error> {
    let mut context = tera::Context::new();
    
    // 添加必要的变量到上下文中
    context.insert("title", "Chat Interface"); // 提供一个默认标题
    context.insert("message", "Hello from Rust and Actix-web!");

    // 渲染模板
    let rendered = tera.render("chat.html", &context)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;

    Ok(HttpResponse::Ok().content_type("text/html").body(rendered))
}

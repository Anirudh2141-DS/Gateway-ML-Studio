use std::{net::SocketAddr, time::Instant};
use axum::{routing::get, Router};
use metrics_exporter_prometheus::PrometheusBuilder;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).init();
    let recorder = PrometheusBuilder::new().install_recorder().unwrap();
    let app = Router::new()
        .route("/up", get(|| async { axum::Json(serde_json::json!({"ok": true})) }))
        .route("/metrics", get(move || {
            let handle = recorder.handle();
            async move { axum::response::Html(handle.render()) }
        }));
    let addr: SocketAddr = "0.0.0.0:9910".parse().unwrap();
    tracing::info!("listening on {}", addr);
    axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app).await.unwrap();
}

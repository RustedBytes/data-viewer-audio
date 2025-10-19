use axum::{
    body,
    extract::{Path as AxumPath, Query, State},
    http,
    response::{self, Html},
    routing::{Router, get},
};
use clap::Parser;
use polars::prelude::*;
use serde::Deserialize;
use std::{
    fs::{self, File},
    io::BufReader,
    path::{Path, PathBuf},
};
use tokio::net::TcpListener;
use tokio_util::io;

/// Command-line arguments for the application.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the folder containing Parquet files.
    folder: String,
    /// Path to the folder containing temp extracted files
    tmp_folder: String,
    /// The address to bind the server to.
    #[arg(short, long, default_value = "0.0.0.0:3000")]
    bind: String,
}

/// Application state shared across handlers.
#[derive(Clone)]
struct AppState {
    folder: PathBuf,
    tmp_folder: PathBuf,
}

/// Represents pagination query parameters.
#[derive(Deserialize, Debug)]
struct Pagination {
    page: Option<usize>,
    page_size: Option<usize>,
}

#[derive(Clone)]
struct Audio {
    path: PathBuf,
    duration: f64,
    transcription: String,
}

fn extract_parquet(path: &Path) -> PolarsResult<DataFrame> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    ParquetReader::new(reader)
        .finish()?
        // Unnest the 'audio' struct column. This creates new columns.
        .unnest(["audio"])
        .map(|mut df| {
            df.rename("bytes", "audio_bytes".into()).unwrap();
            df
        })
        .map(|mut df| {
            df.rename("sampling_rate", "audio_sampling_rate".into())
                .unwrap();
            df
        })
        .map(|mut df| {
            df.rename("path", "audio_path".into()).unwrap();
            df
        })
}

fn extract_parquet_file(tmp_folder: &Path, folder: &Path, filename: &str) -> Vec<Audio> {
    let file_path = folder.join(filename);

    let df = extract_parquet(&file_path).unwrap();

    // Save data frame to temp folder
    let tmp_folder_subdir = tmp_folder.join(filename);

    if !tmp_folder_subdir.exists() {
        fs::create_dir(&tmp_folder_subdir).unwrap();
    }

    let col_d = df.column("duration").unwrap();
    let col_t = df.column("transcription").unwrap();

    let col = df.column("audio_bytes").unwrap();
    let binary_arr = col.binary().unwrap();

    let mut created_files = vec![];

    for i in 0..df.height() {
        let path = tmp_folder_subdir.join(format!("{}.wav", i));

        if !path.exists() {
            let audio_bytes = binary_arr.get(i).unwrap().to_vec();
            let mut file = File::create(path.clone()).unwrap();
            std::io::copy(&mut &audio_bytes[..], &mut file).unwrap();
        }

        let duration = col_d.get(i).unwrap().extract::<f64>().unwrap();
        let transcription = col_t.get(i).unwrap().to_string();

        let audio = Audio {
            path,
            duration,
            transcription,
        };

        created_files.push(audio);
    }

    created_files
}

/// Serves the list of Parquet files in the folder.
async fn list_files(State(state): State<AppState>) -> Html<String> {
    let files: Vec<String> = fs::read_dir(&state.folder)
        .unwrap_or_else(|_| fs::read_dir(".").unwrap()) // Fallback to current directory if specified folder is invalid
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|s| s.to_str()) == Some("parquet"))
        .filter_map(|entry| entry.file_name().to_str().map(|s| s.to_string()))
        .collect();

    let list_items: String = files
        .iter()
        .map(|f| {
            format!(
                r#"<li><a href="/view/{}" class="text-blue-600 hover:underline">{}</a></li>"#,
                f, f
            )
        })
        .collect();

    let html = format!(
        r#"
<!DOCTYPE html>
<html lang="en" class="">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parquet Files</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            darkMode: 'class',
    }}
    </script>
    <script>
        if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {{
            document.documentElement.classList.add('dark')
        }} else {{
            document.documentElement.classList.remove('dark')
        }}
        function toggleTheme() {{
            if (localStorage.theme === 'dark') {{
                localStorage.theme = 'light';
                document.documentElement.classList.remove('dark');
            }} else {{
                localStorage.theme = 'dark';
                document.documentElement.classList.add('dark');
            }}
        }}
    </script>
</head>
<body class="bg-gray-100 dark:bg-gray-900 p-8 text-gray-900 dark:text-gray-100">
    <div class="max-w-4xl mx-auto bg-white dark:bg-gray-800 shadow-md rounded-lg p-6 relative">
        <button onclick="toggleTheme()" class="absolute top-4 right-4 px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded-md text-sm">
            Toggle Theme
        </button>
        <h1 class="text-2xl font-bold mb-4">Parquet Files</h1>
        <ul class="list-disc pl-5 space-y-2">
            {}
        </ul>
    </div>
</body>
</html>
"#,
        list_items
    );

    Html(html)
}

/// Serves a paginated view of the Parquet file data.
async fn view_file(
    State(state): State<AppState>,
    AxumPath(filename): AxumPath<String>,
    Query(pagination): Query<Pagination>,
) -> Html<String> {
    if !filename.ends_with(".parquet") {
        return Html("Invalid file type".to_string());
    }

    let path = state.folder.join(&filename);
    if !path.exists() || !path.is_file() {
        return Html("File not found".to_string());
    }

    let files = extract_parquet_file(&state.tmp_folder, &state.folder, &filename);

    let page = pagination.page.unwrap_or(1);
    let page_size = pagination.page_size.unwrap_or(10);
    let total_items = files.len();
    let total_pages = (total_items as f64 / page_size as f64).ceil() as usize;

    let start = (page - 1) * page_size;
    let end = (start + page_size).min(total_items);

    let paginated_files = if start < files.len() {
        &files[start..end]
    } else {
        &[]
    };
    let mut rows = String::new();
    for audio in paginated_files {
        let audio_src = format!(
            "/audio/{}/{}",
            filename,
            audio.path.file_stem().unwrap().to_str().unwrap()
        );
        rows.push_str(&format!(
            r#"
            <tr class="border-b dark:border-gray-700">
                <td class="px-4 py-2"><audio controls src="{}"></audio></td>
                <td class="px-4 py-2">{}</td>
                <td class="px-4 py-2">{}</td>
            </tr>
            "#,
            audio_src, audio.duration, audio.transcription,
        ));
    }

    let mut pagination_links = String::new();
    for i in 1..=total_pages {
        let class = if i == page {
            "px-3 py-1 bg-blue-500 text-white rounded-md"
        } else {
            "px-3 py-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-blue-600 dark:text-blue-300 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-md"
        };
        pagination_links.push_str(&format!(
            r#"<a href="/view/{}?page={}&page_size={}" class="{}">{}</a>"#,
            filename, i, page_size, class, i
        ));
    }

    let pagination_html = if total_pages > 1 {
        pagination_links
    } else {
        String::new()
    };

    let html = format!(
        r#"
<!DOCTYPE html>
<html lang="en" class="">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - Parquet Viewer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            darkMode: 'class',
    }}
    </script>
    <script>
        if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {{
            document.documentElement.classList.add('dark')
        }} else {{
            document.documentElement.classList.remove('dark')
        }}
        function toggleTheme() {{
            if (localStorage.theme === 'dark') {{
                localStorage.theme = 'light';
                document.documentElement.classList.remove('dark');
            }} else {{
                localStorage.theme = 'dark';
                document.documentElement.classList.add('dark');
            }}
        }}
    </script>
</head>
<body class="bg-gray-100 dark:bg-gray-900 p-8 text-gray-900 dark:text-gray-100">
    <div class="max-w-6xl mx-auto bg-white dark:bg-gray-800 shadow-md rounded-lg p-6 relative">
        <div class="flex justify-between items-center mb-4">
            <a href="/" class="text-blue-600 dark:text-blue-400 hover:underline">Back to list</a>
            <button onclick="toggleTheme()" class="px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded-md text-sm">
                Toggle Theme
            </button>
        </div>
        <h1 class="text-2xl font-bold mb-4">{}</h1>
        <table class="min-w-full bg-white dark:bg-gray-800 border-collapse">
            <thead>
                <tr class="border-b-2 dark:border-gray-700">
                    <th class="px-4 py-2 text-left">Audio</th>
                    <th class="px-4 py-2 text-left">Duration</th>
                    <th class="px-4 py-2 text-left">Transcription</th>
                </tr>
            </thead>
            <tbody>
                {}
            </tbody>
        </table>
        <div class="mt-4 flex flex-wrap justify-center gap-2">
            {}
        </div>
    </div>
</body>
</html>
"#,
        filename, filename, rows, pagination_html
    );

    Html(html)
}

/// Serves audio files from the temporary folder.
async fn serve_audio(
    State(state): State<AppState>,
    AxumPath((filename, index)): AxumPath<(String, String)>,
) -> Result<response::Response, http::StatusCode> {
    let audio_path = state
        .tmp_folder
        .join(&filename)
        .join(format!("{}.wav", index));

    if !audio_path.exists() || !audio_path.is_file() {
        return Err(http::StatusCode::NOT_FOUND);
    }

    let file = tokio::fs::File::open(&audio_path)
        .await
        .map_err(|_| http::StatusCode::INTERNAL_SERVER_ERROR)?;

    let stream = io::ReaderStream::new(file);
    let body = body::Body::from_stream(stream);

    Ok(response::Response::builder()
        .header("Content-Type", "audio/wav")
        .body(body)
        .unwrap())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let folder = PathBuf::from(args.folder);
    if !folder.exists() || !folder.is_dir() {
        return Err("Provided folder does not exist or is not a directory".into());
    }

    let tmp_folder = PathBuf::from(args.tmp_folder.clone());
    if tmp_folder.exists() && tmp_folder.is_dir() {
        fs::remove_dir_all(&tmp_folder)?;
    }
    fs::create_dir_all(&tmp_folder)?;
    if !tmp_folder.exists() || !tmp_folder.is_dir() {
        return Err("Provided tmp_folder does not exist or is not a directory".into());
    }

    let state = AppState { folder, tmp_folder };

    let app = Router::new()
        .route("/", get(list_files))
        .route("/view/{filename}", get(view_file))
        .route("/audio/{filename}/{index}", get(serve_audio))
        .with_state(state);

    println!("Listening on {}", args.bind);

    let listener = TcpListener::bind(&args.bind).await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
}

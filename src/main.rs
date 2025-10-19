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

    let reader_pq = ParquetReader::new(reader);
    reader_pq
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

/// A simple text-based histogram for f64 values, rendered as a string using ASCII bars.
struct Histogram {
    bins: Vec<(f64, f64, usize)>, // (start, end, count)
    max_count: usize,
    bar_width: usize,
    bar_char: char,
}

impl Histogram {
    fn new(values: &[f64], num_bins: usize, bar_width: usize, bar_char: char) -> Self {
        assert!(
            !values.is_empty(),
            "Cannot create histogram from empty data"
        );
        assert!(num_bins > 0, "Number of bins must be greater than 0");

        let min = *values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max = *values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let bin_width = if max == min {
            1.0
        } else {
            (max - min) / num_bins as f64
        };

        let mut bin_counts = vec![0usize; num_bins];
        for &value in values {
            if value < min || value > max {
                continue; // Skip outliers if any, though unlikely
            }
            let bin_idx = ((value - min) / bin_width).min((num_bins - 1) as f64) as usize;
            bin_counts[bin_idx] += 1;
        }

        let max_count = *bin_counts.iter().max().unwrap_or(&0);

        let mut bins = Vec::new();
        for (i, &count) in bin_counts.iter().enumerate() {
            let start = min + (i as f64 * bin_width);
            let end = if i == num_bins - 1 {
                max
            } else {
                start + bin_width
            };
            bins.push((start, end, count));
        }

        Self {
            bins,
            max_count,
            bar_width,
            bar_char,
        }
    }

    /// Renders the histogram as a formatted string.
    fn render(&self, field: &str) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Histogram of {}: {} values\n",
            field,
            self.bins.iter().map(|b| b.2).sum::<usize>()
        ));
        output.push_str("Bin Range\t\tFrequency\n");
        output.push_str(&"-".repeat(40));
        output.push('\n');

        for (start, end, count) in &self.bins {
            let bar_length = if self.max_count > 0 {
                ((*count as f64 / self.max_count as f64) * self.bar_width as f64).round() as usize
            } else {
                0
            };
            let bar = std::iter::repeat_n(self.bar_char, bar_length).collect::<String>();
            let range_str = format!("[{:.2} - {:.2})", start, end);
            output.push_str(&format!("{}\t{:>8}\t{}\n", range_str, count, bar));
        }

        output
    }
}

fn plot_durations(data: &[f64]) -> String {
    let hist = Histogram::new(data, 4, 20, '*');

    hist.render("durations")
}

/// A simple text-based histogram for integer values, rendered as a string using ASCII bars.
struct IntHistogram {
    bins: Vec<(usize, usize, usize)>, // (start, end, count)
    max_count: usize,
    bar_width: usize,
    bar_char: char,
}

impl IntHistogram {
    fn new(values: &[usize], num_bins: usize, bar_width: usize, bar_char: char) -> Self {
        assert!(
            !values.is_empty(),
            "Cannot create histogram from empty data"
        );
        assert!(num_bins > 0, "Number of bins must be greater than 0");

        let min = *values.iter().min().unwrap();
        let max = *values.iter().max().unwrap();

        let bin_width = if max == min {
            1
        } else {
            // Ensure bin_width is at least 1
            ((max - min) as f64 / num_bins as f64).ceil() as usize
        };

        let mut bin_counts = vec![0usize; num_bins];
        for &value in values {
            if value < min || value > max {
                continue;
            }
            let bin_idx = if bin_width > 0 {
                ((value - min) / bin_width).min(num_bins - 1)
            } else {
                0
            };
            bin_counts[bin_idx] += 1;
        }

        let max_count = *bin_counts.iter().max().unwrap_or(&0);

        let mut bins = Vec::new();
        for (i, &count) in bin_counts.iter().enumerate() {
            let start = min + (i * bin_width);
            let end = start + bin_width;
            bins.push((start, end, count));
        }

        Self {
            bins,
            max_count,
            bar_width,
            bar_char,
        }
    }

    /// Renders the histogram as a formatted string.
    fn render(&self, field: &str) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Histogram of {}: {} values\n",
            field,
            self.bins.iter().map(|b| b.2).sum::<usize>()
        ));
        output.push_str("Bin Range\t\tFrequency\n");
        output.push_str(&"-".repeat(40));
        output.push('\n');

        for (start, end, count) in &self.bins {
            let bar_length = if self.max_count > 0 {
                ((*count as f64 / self.max_count as f64) * self.bar_width as f64).round() as usize
            } else {
                0
            };
            let bar = std::iter::repeat_n(self.bar_char, bar_length).collect::<String>();
            let range_str = format!("[{} - {})", start, end);
            output.push_str(&format!("{}\t{:>8}\t{}\n", range_str, count, bar));
        }
        output
    }
}

fn plot_transcription_lengths(data: &[usize]) -> String {
    let hist = IntHistogram::new(data, 4, 20, '*');
    hist.render("transcription lengths")
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
        let transcription = if let AnyValue::String(s) = col_t.get(i).unwrap() {
            s.to_string()
        } else {
            col_t.get(i).unwrap().to_string()
        };

        let audio = Audio {
            path,
            duration,
            transcription,
        };

        created_files.push(audio);
    }

    created_files
}

/// Formats a duration in seconds into a human-readable string (MM:SS.ms or HH:MM:SS.ms).
fn format_duration(seconds: f64) -> String {
    let total_seconds = seconds.floor() as u64;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let secs = total_seconds % 60;
    let millis = (seconds.fract() * 1000.0).round() as u64;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, millis)
    } else {
        format!("{:02}:{:02}.{:03}", minutes, secs, millis)
    }
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
    <footer class="text-center mt-4">
        <a href="https://github.com/RustedBytes/data-viewer-audio" class="text-sm text-gray-500 dark:text-gray-400 hover:underline"><b>data-viewer-audio</b> on GitHub</a>
    </footer>
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
            <tr class="border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer" onclick="var audio = this.querySelector('audio'); if (audio.paused) {{ audio.play(); }} else {{ audio.pause(); }}">
                <td class="px-4 py-4"><audio class="h-dvh max-h-[2.25rem] w-full min-w-[300px] max-w-xs" controls="" preload="none">
                    <source src="{}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </td>
                <td class="px-4 py-4 text-right">{}</td>
                <td class="px-4 py-4">{}</td>
            </tr>
            "#,
            audio_src, format_duration(audio.duration), &audio.transcription,
        ));
    }

    let pagination_html = if total_pages > 1 {
        let mut pagination_links = String::new();
        let window = 2;
        let mut pages_to_render = vec![];

        // Previous page link
        if page > 1 {
            pagination_links.push_str(&format!(
                r#"<a href="/view/{}?page={}&page_size={}" class="px-3 py-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-blue-600 dark:text-blue-300 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-md">Prev</a>"#,
                filename, page - 1, page_size
            ));
        }

        pages_to_render.push(1);
        if page > window + 2 {
            pages_to_render.push(0); // Ellipsis
        }

        for i in (page.saturating_sub(window))..=(page + window) {
            if i > 1 && i < total_pages {
                pages_to_render.push(i);
            }
        }

        if page < total_pages.saturating_sub(window + 1) {
            pages_to_render.push(0); // Ellipsis
        }
        if total_pages > 1 {
            pages_to_render.push(total_pages);
        }

        pages_to_render.dedup();

        for p in pages_to_render {
            if p == 0 {
                pagination_links.push_str(r#"<span class="px-3 py-1">...</span>"#);
            } else {
                let class = if p == page {
                    "px-3 py-1 bg-blue-500 text-white rounded-md"
                } else {
                    "px-3 py-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-blue-600 dark:text-blue-300 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-md"
                };
                pagination_links.push_str(&format!(
                    r#"<a href="/view/{}?page={}&page_size={}" class="{}">{}</a>"#,
                    filename, p, page_size, class, p
                ));
            }
        }

        // Next page link
        if page < total_pages {
            pagination_links.push_str(&format!(r#"<a href="/view/{}?page={}&page_size={}" class="px-3 py-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-blue-600 dark:text-blue-300 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-md">Next</a>"#, filename, page + 1, page_size));
        }
        pagination_links
    } else {
        String::new()
    };

    let page_size_selector = {
        let sizes = [10, 25, 50, 100];
        let mut options = String::new();
        for &size in &sizes {
            let selected = if size == page_size { "selected" } else { "" };
            options.push_str(&format!(
                r#"<option value="/view/{}?page=1&page_size={}" {}>{}</option>"#,
                filename, size, selected, size
            ));
        }

        format!(
            r#"<div class="flex items-center gap-2"><span>Page size:</span><select onchange="location = this.value;" class="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 rounded-md p-1">{}</select></div>"#,
            options
        )
    };

    let durations: Vec<f64> = files.iter().map(|a| a.duration).collect();
    let durations_plot = plot_durations(&durations);

    let transcriptions: Vec<usize> = files.iter().map(|a| a.transcription.len()).collect();
    let transcriptions_plot = plot_transcription_lengths(&transcriptions);

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
    <script>
        document.addEventListener('play', function(e) {{
            var audios = document.getElementsByTagName('audio');
            for (var i = 0, len = audios.length; i < len; i++) {{
                if (audios[i] != e.target) {{
                    audios[i].pause();
                }}
            }}
        }}, true);
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
        <details class="mb-4 bg-gray-50 dark:bg-gray-700 p-4 rounded">
            <summary class="font-semibold cursor-pointer">Metadata details</summary>
            <pre class="mt-2 text-sm text-gray-600 dark:text-gray-300 whitespace-pre-wrap"><code>{}</code></pre>
            <br>
            <pre class="mt-2 text-sm text-gray-600 dark:text-gray-300 whitespace-pre-wrap"><code>{}</code></pre>
        </details>
        <table class="min-w-full bg-white dark:bg-gray-800 border-collapse">
            <thead>
                <tr class="border-b-2 dark:border-gray-700">
                    <th class="px-4 py-2 text-left">Audio</th>
                    <th class="px-4 py-2 text-right">Duration</th>
                    <th class="px-4 py-2 text-left">Transcription</th>
                </tr>
            </thead>
            <tbody>
                {}
            </tbody>
        </table>
        <div class="mt-4 flex flex-col items-center gap-4">
            <div class="flex flex-wrap justify-center gap-2">
                {}
            </div>
            <div class="flex flex-wrap justify-center gap-2">
                {}
            </div>
            <div class="text-center text-sm text-gray-500 dark:text-gray-400">
                Total audio files: {}
            </div>
            <footer class="text-center mt-4">
                <a href="https://github.com/RustedBytes/data-viewer-audio" class="text-sm text-gray-500 dark:text-gray-400 hover:underline"><b>data-viewer-audio</b> on GitHub</a>
            </footer>
        </div>
    </div>
</body>
</html>
"#,
        filename,
        filename,
        durations_plot,
        transcriptions_plot,
        rows,
        pagination_html,
        page_size_selector,
        total_items
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

    println!("Listening on http://{}", args.bind);

    let listener = TcpListener::bind(&args.bind).await.unwrap();
    axum::serve(listener, app).await.unwrap();

    Ok(())
}

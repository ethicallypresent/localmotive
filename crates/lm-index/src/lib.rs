//! The LocalMotive SQLite index.
//!
//! * `chunks` — every indexed chunk (HTML chunks plus visible OCR chunks).
//! * `chunk_dtc` — exact DTC code → chunk lookup.
//! * `chunks_fts` — FTS5 (`porter unicode61`) over title, breadcrumb, text and
//!   the chunk's DTC codes (`P0300` is one token, so codes match exactly).
//! * `vec_chunks` — sqlite-vec `vec0` table, `float[1024]` for
//!   `qwen3-embedding:0.6b`; filled by `lm-index embed` once Ollama has the model.
//!
//! `rowid` of `chunks_fts` and `vec_chunks` equals `chunks.rid`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Once;

use anyhow::{Context, Result, bail};
use lm_types::{Chunk, Kind, Source, approx_tokens};
use rusqlite::{Connection, OptionalExtension, Row, params};

pub const SCHEMA_VERSION: u32 = 1;
pub const EMBED_MODEL: &str = "qwen3-embedding:0.6b";
pub const EMBED_DIM: usize = 1024;

pub fn default_db_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join("localmotive/data/localmotive.db")
}

pub fn default_corpus_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join("localmotive/corpus")
}

static VEC_INIT: Once = Once::new();

/// Register sqlite-vec for every connection opened afterwards.
fn register_vec() {
    VEC_INIT.call_once(|| {
        // SAFETY: sqlite3_vec_init has the signature SQLite expects of an
        // extension entry point; registering it as an auto-extension is the
        // documented way to load sqlite-vec with a bundled SQLite.
        unsafe {
            #[allow(clippy::missing_transmute_annotations)]
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

pub fn open(path: &Path) -> Result<Connection> {
    register_vec();
    let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
    Ok(conn)
}

pub fn open_readonly(path: &Path) -> Result<Connection> {
    register_vec();
    if !path.exists() {
        bail!(
            "index {} not found; run `lm-index build` first",
            path.display()
        );
    }
    Ok(Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?)
}

pub fn create_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(&format!(
        "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
         CREATE TABLE chunks(
           rid INTEGER PRIMARY KEY,
           id TEXT NOT NULL UNIQUE,
           page_id INTEGER NOT NULL,
           title TEXT NOT NULL,
           breadcrumb TEXT NOT NULL,
           aliases TEXT NOT NULL,          -- JSON array
           section_heading TEXT NOT NULL,
           text TEXT NOT NULL,
           images TEXT NOT NULL,           -- JSON array
           links TEXT NOT NULL,            -- JSON array
           kind TEXT NOT NULL,
           dtc_codes TEXT NOT NULL,        -- JSON array
           image_only INTEGER NOT NULL,
           source TEXT NOT NULL,
           ocr_confidence REAL,
           ocr_raw TEXT,
           ocr_checked_rows TEXT NOT NULL, -- JSON array of line indexes
           ocr_masked_values INTEGER NOT NULL,
           hidden INTEGER NOT NULL,
           tokens INTEGER NOT NULL
         );
         CREATE INDEX chunks_page ON chunks(page_id);
         CREATE INDEX chunks_kind ON chunks(kind);
         CREATE TABLE chunk_dtc(code TEXT NOT NULL, rid INTEGER NOT NULL, PRIMARY KEY(code, rid)) WITHOUT ROWID;
         CREATE VIRTUAL TABLE chunks_fts USING fts5(title, breadcrumb, text, dtc, tokenize = 'porter unicode61');
         CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{EMBED_DIM}]);"
    ))?;
    Ok(())
}

pub fn load_jsonl(path: &Path) -> Result<Vec<Chunk>> {
    let s = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(l).with_context(|| format!("{}:{}", path.display(), i + 1))
        })
        .collect()
}

fn json<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "[]".into())
}

fn source_str(s: Source) -> &'static str {
    match s {
        Source::Html => "html",
        Source::Ocr => "ocr",
    }
}

pub fn insert_chunks(conn: &mut Connection, chunks: &[Chunk]) -> Result<()> {
    let tx = conn.transaction()?;
    {
        let mut ins = tx.prepare(
            "INSERT INTO chunks(id, page_id, title, breadcrumb, aliases, section_heading, text, images, links, kind,
                                dtc_codes, image_only, source, ocr_confidence, ocr_raw, ocr_checked_rows, ocr_masked_values,
                                hidden, tokens)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
        )?;
        let mut fts = tx.prepare("INSERT INTO chunks_fts(rowid, title, breadcrumb, text, dtc) VALUES (?1, ?2, ?3, ?4, ?5)")?;
        let mut dtc = tx.prepare("INSERT OR IGNORE INTO chunk_dtc(code, rid) VALUES (?1, ?2)")?;
        for c in chunks {
            ins.execute(params![
                c.id,
                c.page_id,
                c.title,
                c.breadcrumb,
                json(&c.aliases),
                c.section_heading,
                c.text,
                json(&c.images),
                json(&c.links),
                c.kind.as_str(),
                json(&c.dtc_codes),
                c.image_only,
                source_str(c.source),
                c.ocr_confidence,
                c.ocr_raw,
                json(&c.ocr_checked_rows),
                c.ocr_masked_values,
                c.hidden,
                approx_tokens(&c.text) as i64,
            ])?;
            let rid = tx.last_insert_rowid();
            fts.execute(params![
                rid,
                c.title,
                c.breadcrumb,
                c.text,
                c.dtc_codes.join(" ")
            ])?;
            for code in &c.dtc_codes {
                dtc.execute(params![code, rid])?;
            }
        }
    }
    tx.commit()?;
    Ok(())
}

pub fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?1, ?2)",
        params![key, value],
    )?;
    Ok(())
}

pub fn meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
        .optional()?)
}

pub struct BuildStats {
    pub html: usize,
    pub ocr_visible: usize,
    pub ocr_hidden: usize,
}

/// Build a fresh index from `corpus/chunks.jsonl` + visible `chunks.ocr.jsonl`.
/// Writes to a temp file and renames it over `db` when complete.
pub fn build(corpus: &Path, db: &Path) -> Result<BuildStats> {
    let html = load_jsonl(&corpus.join("chunks.jsonl"))?;
    let ocr_path = corpus.join("chunks.ocr.jsonl");
    let ocr = if ocr_path.exists() {
        load_jsonl(&ocr_path)?
    } else {
        Vec::new()
    };
    let (visible, hidden): (Vec<Chunk>, Vec<Chunk>) = ocr.into_iter().partition(|c| !c.hidden);

    if let Some(dir) = db.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = db.with_extension("db.tmp");
    let _ = fs::remove_file(&tmp);
    let mut conn = open(&tmp)?;
    create_schema(&conn)?;
    let mut all = html.clone();
    all.extend(visible.iter().cloned());
    insert_chunks(&mut conn, &all)?;
    for (k, v) in [
        ("schema_version", SCHEMA_VERSION.to_string()),
        (
            "built_at_unix",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
                .to_string(),
        ),
        ("chunks_html", html.len().to_string()),
        ("chunks_ocr", visible.len().to_string()),
        ("embed_model", EMBED_MODEL.to_string()),
        ("embed_dim", EMBED_DIM.to_string()),
    ] {
        set_meta(&conn, k, &v)?;
    }
    conn.execute_batch("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize'); VACUUM;")?;
    drop(conn);
    fs::rename(&tmp, db)?;
    Ok(BuildStats {
        html: html.len(),
        ocr_visible: visible.len(),
        ocr_hidden: hidden.len(),
    })
}

fn parse_kind(s: &str) -> Kind {
    Kind::ALL
        .into_iter()
        .find(|k| k.as_str() == s)
        .unwrap_or(Kind::Other)
}

pub const CHUNK_COLUMNS: &str = "rid, id, page_id, title, breadcrumb, aliases, section_heading, text, images, links, kind, \
                                 dtc_codes, image_only, source, ocr_confidence, ocr_raw, ocr_checked_rows, ocr_masked_values, hidden";

/// Map a row selected with [`CHUNK_COLUMNS`] to (rid, Chunk).
pub fn row_to_chunk(r: &Row) -> rusqlite::Result<(i64, Chunk)> {
    let arr = |i: usize| -> rusqlite::Result<Vec<String>> {
        Ok(serde_json::from_str(&r.get::<_, String>(i)?).unwrap_or_default())
    };
    let source: String = r.get(13)?;
    Ok((
        r.get(0)?,
        Chunk {
            id: r.get(1)?,
            page_id: r.get(2)?,
            title: r.get(3)?,
            breadcrumb: r.get(4)?,
            aliases: arr(5)?,
            section_heading: r.get(6)?,
            text: r.get(7)?,
            images: arr(8)?,
            links: serde_json::from_str(&r.get::<_, String>(9)?).unwrap_or_default(),
            kind: parse_kind(&r.get::<_, String>(10)?),
            dtc_codes: arr(11)?,
            image_only: r.get(12)?,
            source: if source == "ocr" {
                Source::Ocr
            } else {
                Source::Html
            },
            ocr_confidence: r.get(14)?,
            ocr_raw: r.get(15)?,
            ocr_checked_rows: serde_json::from_str(&r.get::<_, String>(16)?).unwrap_or_default(),
            ocr_masked_values: r.get(17)?,
            hidden: r.get(18)?,
        },
    ))
}

pub fn chunk_by_rid(conn: &Connection, rid: i64) -> Result<Option<Chunk>> {
    Ok(conn
        .query_row(
            &format!("SELECT {CHUNK_COLUMNS} FROM chunks WHERE rid = ?1"),
            [rid],
            row_to_chunk,
        )
        .optional()?
        .map(|(_, c)| c))
}

pub fn chunk_by_id(conn: &Connection, id: &str) -> Result<Option<Chunk>> {
    Ok(conn
        .query_row(
            &format!("SELECT {CHUNK_COLUMNS} FROM chunks WHERE id = ?1"),
            [id],
            row_to_chunk,
        )
        .optional()?
        .map(|(_, c)| c))
}

pub fn vec_count(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("SELECT count(*) FROM vec_chunks", [], |r| r.get(0))?)
}

pub fn f32_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Embed every chunk that has no vector yet. Needs Ollama with [`EMBED_MODEL`].
pub async fn embed_missing(
    conn: &Connection,
    client: &lm_ollama::Client,
    batch: usize,
) -> Result<usize> {
    client
        .version()
        .await
        .context("Ollama is not running on 127.0.0.1:11434 (start it with `ollama serve`)")?;
    if !client.has_model(EMBED_MODEL).await? {
        bail!("embedding model {EMBED_MODEL} is not pulled; run: ollama pull {EMBED_MODEL}");
    }
    let todo: Vec<(i64, Chunk)> = {
        let mut st = conn.prepare(&format!(
            "SELECT {CHUNK_COLUMNS} FROM chunks WHERE rid NOT IN (SELECT rowid FROM vec_chunks) ORDER BY rid"
        ))?;
        st.query_map([], row_to_chunk)?
            .collect::<rusqlite::Result<_>>()?
    };
    let mut done = 0;
    for part in todo.chunks(batch.max(1)) {
        let texts: Vec<String> = part.iter().map(|(_, c)| c.context_text()).collect();
        let vecs = client.embed(EMBED_MODEL, &texts, Some("10m")).await?;
        for ((rid, _), v) in part.iter().zip(vecs) {
            if v.len() != EMBED_DIM {
                bail!(
                    "{EMBED_MODEL} returned {} dims, index expects {EMBED_DIM}",
                    v.len()
                );
            }
            conn.execute(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES (?1, ?2)",
                params![rid, f32_blob(&v)],
            )?;
        }
        done += part.len();
        eprintln!("embedded {done}/{}", todo.len());
    }
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, title: &str, text: &str, codes: &[&str]) -> Chunk {
        Chunk {
            id: id.into(),
            page_id: 1,
            title: title.into(),
            breadcrumb: "A > B".into(),
            aliases: vec!["A > B".into()],
            section_heading: String::new(),
            text: text.into(),
            images: vec![],
            links: vec![9453],
            kind: Kind::Dtc,
            dtc_codes: codes.iter().map(|s| s.to_string()).collect(),
            image_only: false,
            source: Source::Html,
            ocr_confidence: None,
            ocr_raw: None,
            ocr_checked_rows: Vec::new(),
            ocr_masked_values: 0,
            hidden: false,
        }
    }

    #[test]
    fn schema_insert_fts_dtc_and_vec() {
        let mut conn = open(Path::new(":memory:")).unwrap();
        create_schema(&conn).unwrap();
        let cs = vec![
            chunk("a", "P0300", "The PCM detects engine misfires.", &["P0300"]),
            chunk(
                "b",
                "Spark Plug Gap",
                "Spark Plug Gap 1.52 mm (0.060 in.)",
                &[],
            ),
        ];
        insert_chunks(&mut conn, &cs).unwrap();
        let hit: String = conn
            .query_row("SELECT c.id FROM chunks_fts f JOIN chunks c ON c.rid = f.rowid WHERE chunks_fts MATCH '\"p0300\"'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(hit, "a");
        // porter stemming: "misfiring" matches "misfires"
        let n: i64 = conn
            .query_row(
                "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'misfiring'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1);
        let rid: i64 = conn
            .query_row("SELECT rid FROM chunk_dtc WHERE code = 'P0300'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(chunk_by_rid(&conn, rid).unwrap().unwrap(), cs[0]);
        assert_eq!(chunk_by_id(&conn, "b").unwrap().unwrap().links, vec![9453]);
        // sqlite-vec is loaded and the vec0 table takes 1024-d vectors
        let v = vec![0.5f32; EMBED_DIM];
        conn.execute(
            "INSERT INTO vec_chunks(rowid, embedding) VALUES (?1, ?2)",
            params![rid, f32_blob(&v)],
        )
        .unwrap();
        assert_eq!(vec_count(&conn).unwrap(), 1);
        let (got, dist): (i64, f64) = conn
            .query_row("SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ?1 ORDER BY distance LIMIT 1", [f32_blob(&v)], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!((got, dist), (rid, 0.0));
    }
}
